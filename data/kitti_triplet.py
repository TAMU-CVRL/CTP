import os
import json
import tarfile
import io
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.data_utils import crop_annotation_kitti, compute_box_corners, points_in_3d_box
from utils.pc_utils import segment_ground_o3d

class KITTI_TripletDataset(Dataset):
    def __init__(self, root_dir, split='training', min_points=5):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, 'image_2')
        self.velo_dir = os.path.join(self.root_dir, 'velodyne')
        self.label_dir = os.path.join(self.root_dir, 'label_2')
        self.indices = [f.split('.')[0] for f in sorted(os.listdir(self.image_dir)) if f.endswith('.png')]
        self.min_points = min_points

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx_str = self.indices[idx]
        img_path = os.path.join(self.image_dir, f"{idx_str}.png")
        lidar_path = os.path.join(self.velo_dir, f"{idx_str}.bin")
        label_path = os.path.join(self.label_dir, f"{idx_str}.txt")
        calib_path = os.path.join(self.root_dir, "calib", f"{idx_str}.txt")

        # Read Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read Point Cloud
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        _, lidar_non_ground = segment_ground_o3d(lidar)
        # Read Labels
        labels = self.read_labels(label_path)
        # Read Calibration Matrix
        Tr_velo_to_cam, R0_rect = load_calib(calib_path)
        # Generate triplets (passing calibration parameters)
        til_triplet, all_bboxes = self.prepare_triplet(image, lidar_non_ground, labels, Tr_velo_to_cam, R0_rect)

        return {
            "til_triplet": til_triplet,
            "all_bboxes": all_bboxes
        }

    def read_labels(self, path):
        """Read KITTI label_2 file"""
        labels = []
        with open(path, 'r') as f:
            for line in f.readlines():
                obj = line.strip().split(' ')
                if len(obj) < 15:
                    continue
                cls = obj[0]
                h, w, l = map(float, obj[8:11])
                x, y, z = map(float, obj[11:14])
                ry = float(obj[14])
                bbox_2d = list(map(float, obj[4:8]))
                labels.append({
                    "class": cls,
                    "h": h, "w": w, "l": l,
                    "x": x, "y": y, "z": z,
                    "ry": ry,
                    "bbox_2d": bbox_2d
                })
        return labels

    def prepare_triplet(self, image, lidar, labels, Tr_velo_to_cam, R0_rect):
        til_triplet, all_bboxes = [], []
        image_pil = Image.fromarray(image)

        for ann in labels:
            # Coordinate Transformation
            x, y, z, h, w, l, ry = camera_box_to_lidar(
                ann["x"], ann["y"], ann["z"], ann["h"], ann["w"], ann["l"], ann["ry"],
                Tr_velo_to_cam, R0_rect
            )

            box_corners = compute_box_corners(x, y, z, h, w, l, ry)
            mask = points_in_3d_box(box_corners, lidar[:, :3])
            lidar_obj = lidar[mask]
            if len(lidar_obj) < self.min_points:
                continue

            crop = crop_annotation_kitti(image_pil, ann["bbox_2d"], margin=5)
            if crop is None:
                continue

            bbox_info = [x, y, z, h, w, l, ry]
            all_bboxes.append({"bbox": bbox_info, "category": ann["class"]})
            til_triplet.append((ann["class"], crop, lidar_obj, bbox_info))

        return til_triplet, all_bboxes

class Triplet_Object_KITTI(Dataset):
    def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, prompt=""):
        # 1. Automatic Path Inference
        jsonl_path = Path(jsonl_file).resolve()
        self.data_root = jsonl_path.parent  # Usually dataset/kitti_triplets/
        
        # 2. Load JSONL Records
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_transform = image_transform
        self.sparse_to_dense_fn = sparse_to_dense_fn
        self.prompt = prompt
        
        # 3. Archive Management
        self.image_tar = None
        self.lidar_tar = None
        self.image_members = {}
        self.lidar_members = {}

        self._auto_detect_archives()

    def _auto_detect_archives(self):
        """Scans the directory for .tar files and maps their contents."""
        tar_candidates = list(self.data_root.glob("*.tar"))
        
        for tar_path in tar_candidates:
            name_lower = tar_path.name.lower()
            try:
                handle = tarfile.open(tar_path, "r")
                members_map = {}
                
                for m in handle.getmembers():
                    clean_name = m.name
                    # Remove common redundant prefixes like 'data/kitti_images/'
                    if "image" in clean_name:
                        idx = clean_name.find("image")
                        clean_name = clean_name[max(0, clean_name.rfind("/", 0, idx) + 1):]
                    elif "lidar" in clean_name:
                        idx = clean_name.find("lidar")
                        clean_name = clean_name[max(0, clean_name.rfind("/", 0, idx) + 1):]
                    
                    members_map[clean_name] = m

                if "image" in name_lower:
                    self.image_tar = handle
                    self.image_members = members_map
                    print(f"[INFO] KITTI Image TAR detected: {tar_path.name}")
                elif "lidar" in name_lower:
                    self.lidar_tar = handle
                    self.lidar_members = members_map
                    print(f"[INFO] KITTI LiDAR TAR detected: {tar_path.name}")
                else:
                    handle.close()
            except Exception as e:
                print(f"[WARN] Could not process tar {tar_path.name}: {e}")

    def _load_resource(self, rel_path, tar_handle, members_map, is_numpy=False):
        """Unified loader: Checks TAR archive first, then local disk."""
        # Sanitize the relative path to match the member map keys
        # (e.g., if JSONL has 'data/kitti_images/train/x.png', convert to 'kitti_images/train/x.png')
        if "image" in rel_path:
            idx = rel_path.find("image")
            clean_rel = rel_path[max(0, rel_path.rfind("/", 0, idx) + 1):]
        elif "lidar" in rel_path:
            idx = rel_path.find("lidar")
            clean_rel = rel_path[max(0, rel_path.rfind("/", 0, idx) + 1):]
        else:
            clean_rel = rel_path

        # 1. Try Archive
        if tar_handle and clean_rel in members_map:
            member = members_map[clean_rel]
            file_data = tar_handle.extractfile(member).read()
            if is_numpy:
                return np.load(io.BytesIO(file_data))
            return Image.open(io.BytesIO(file_data)).convert("RGB")
        
        # 2. Fallback to local Disk
        # We try both the raw rel_path and the clean_rel relative to data_root
        disk_path = self.data_root / rel_path
        if not disk_path.exists():
            disk_path = self.data_root / clean_rel
            
        if not disk_path.exists():
            raise FileNotFoundError(f"Resource {rel_path} not found in TAR or local folder at {self.data_root}")
        
        if is_numpy:
            return np.load(disk_path)
        return Image.open(disk_path).convert("RGB")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text
        label = self.prompt + item["label"]
        caption = item.get("caption", "")

        # Image
        img = self._load_resource(item["image_path"], self.image_tar, self.image_members)
        img = self.image_transform(img)

        # LiDAR
        if "lidar_path" in item:
            lidar_np = self._load_resource(item["lidar_path"], self.lidar_tar, self.lidar_members, is_numpy=True)
            lidar_data = torch.from_numpy(lidar_np).float()
        else:
            # Handle list of points from JSONL
            lidar_data = torch.tensor(item["lidar"], dtype=torch.float32)
            
        lidar = self.sparse_to_dense_fn(lidar_data)

        return {
            "label": label,
            "caption": caption,
            "image": img,
            "lidar": lidar
        }

    def __del__(self):
        if hasattr(self, 'image_tar') and self.image_tar:
            self.image_tar.close()
        if hasattr(self, 'lidar_tar') and self.lidar_tar:
            self.lidar_tar.close()
