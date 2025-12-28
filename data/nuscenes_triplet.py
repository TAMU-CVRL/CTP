import numpy as np
import io
import json
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.pc_utils import lidar2camera_fov, segment_ground_o3d, zero_pad, load_lidar_bin
import tarfile

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
from utils.img_utils import crop_annotation

class Nuscenes_TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.nusc = base_dataset.nusc
        
    def __len__(self):
        return len(self.base_dataset)  # Return the number of samples in the dataset
       
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]  # Fetch the sample by index
        
        if 'raw_lidar' in sample:
            last_lidar = sample['raw_lidar'][-1, :, :3]  # current frame, [T, max_N, 4] -> [max_N, 3]
            til_triplet, all_bboxes = self.prepare_triplet(sample, last_lidar)
        sample['til_triplet'] = til_triplet
        sample['all_bboxes'] = all_bboxes
        return sample
    
    def prepare_triplet(self, sample, pc_ego):
        """
        sample: a sample from NuScenes
        pc_ego: point cloud in ego coordinate, shape [N, 3] or [N, 4] (x, y, z [, intensity])
        """
        sample_record = self.nusc.get('sample', sample['token'][-1])
        lidar_token = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_token['ego_pose_token'])
        til_triplet = []
        all_bboxes = []

        for ann_token in sample_record['anns']:
            # 1. Check visibility
            ann_record = self.nusc.get('sample_annotation', ann_token)
            visible = int(ann_record['visibility_token'])
            if visible < 2: # 0: unknown, 1: not_visible, 2: partly, 3: fully
                continue
            # Get label
            full_label = ann_record['category_name']
            parts = full_label.split('.')
            # Use the second part as the label if it exists, otherwise use the last part
            if len(parts) >= 2:
                label = parts[1]
            else:
                label = parts[-1]
            # Get 3D bounding box
            box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))

            # Step 1: global -> ego
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            yaw = box.orientation.yaw_pitch_roll[0]
            bbox_inf = [*box.center.tolist(), *box.wlh.tolist(), float(yaw)]
            all_bboxes.append({
                'sample_token': ann_record['token'],
                'instance_token': ann_record['instance_token'],
                'bbox': bbox_inf,
                'category': ann_record['category_name']
            })
            
            # Extract points within the bounding box
            mask = points_in_box(box, pc_ego[:, :3].T)
            points_in_instance = pc_ego[mask]

            # 2. Check the number of points
            if len(points_in_instance) < 5:
                continue
            
            # Crop the image around the bounding box
            cropped_image = crop_annotation(self.nusc, ann_token, sample_record, margin=5)
            # 3. Check the cropped image
            if cropped_image is None:
                continue
            
            # Build JSON-serializable format
            til_triplet.append((label, cropped_image, points_in_instance, bbox_inf))
        return til_triplet, all_bboxes

# class Triplet_Object_Nuscenes(Dataset):
#     def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, image_tar_path = None, prompt="A "):
#         self.data = []
#         with open(jsonl_file, "r") as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#         self.image_transform = image_transform
#         self.sparse_to_dense_fn = sparse_to_dense_fn
#         self.prompt = prompt
#         self.image_tar_path = image_tar_path
#         if self.image_tar_path is not None:
#             # Open tar archive once
#             self.tar = tarfile.open(image_tar_path, "r")
#             self.members = {}
#             for m in self.tar.getmembers():
#                 rel_path = m.name
#                 if rel_path.startswith("data/nuscenes_images/"):
#                     rel_path = rel_path[len("data/nuscenes_images/"):]
#                 self.members[rel_path] = m
#             print(f"[INFO] Images will be loaded from TAR archive: {image_tar_path}")
#         else:
#             print(f"[INFO] Images will be loaded from disk directly.")    

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
        
#         # Text processing
#         label = self.prompt + item["label"]
#         caption = item.get("caption", "")

#         # Image processing
#         if self.image_tar_path is None: # Load image from disk directlyimage
#             img = Image.open(item["image_path"]).convert("RGB")
#         else: # Load image from tar instead of disk
#             img_rel_path = item["image_path"]
#             if img_rel_path.startswith("data/nuscenes_images/"):
#                 img_rel_path = img_rel_path[len("data/nuscenes_images/"):]
            
#             if img_rel_path not in self.members:
#                 raise FileNotFoundError(f"{img_rel_path} not found in tar archive")

#             img_member = self.members[img_rel_path]
#             img_file = self.tar.extractfile(img_member)
#             img = Image.open(io.BytesIO(img_file.read())).convert("RGB")

#         img = self.image_transform(img)

#         # Lidar processing
#         lidar = torch.tensor(item["lidar"])
#         lidar = self.sparse_to_dense_fn(lidar)

#         return {
#             "label": label,
#             "caption": caption,
#             "image": img,
#             "lidar": lidar
#         }

class Triplet_Object_Nuscenes(Dataset):
    def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, prompt=""):
        # 1. Automatic Path Inference
        jsonl_path = Path(jsonl_file).resolve()
        self.data_root = jsonl_path.parent  # nuscenes_triplets
        
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
        # Find all .tar files in the same folder as the JSONL
        tar_candidates = list(self.data_root.glob("*.tar"))
        
        for tar_path in tar_candidates:
            name_lower = tar_path.name.lower()
            try:
                handle = tarfile.open(tar_path, "r")
                members_map = {}
                
                for m in handle.getmembers():
                    # Sanitize paths to match JSONL (e.g., nuscenes_image/train/...)
                    clean_name = m.name
                    # Remove common redundant prefixes like 'data/' or parent folder names
                    if "image" in clean_name:
                        idx = clean_name.find("image")
                        # Capture from the start of the dataset name (e.g., 'nuscenes_image')
                        clean_name = clean_name[max(0, clean_name.rfind("/", 0, idx) + 1):]
                    elif "lidar" in clean_name:
                        idx = clean_name.find("lidar")
                        clean_name = clean_name[max(0, clean_name.rfind("/", 0, idx) + 1):]
                    
                    members_map[clean_name] = m

                # Assign handle based on filename keywords
                if "image" in name_lower:
                    self.image_tar = handle
                    self.image_members = members_map
                    print(f"[INFO] Image TAR detected: {tar_path.name}")
                elif "lidar" in name_lower:
                    self.lidar_tar = handle
                    self.lidar_members = members_map
                    print(f"[INFO] LiDAR TAR detected: {tar_path.name}")
                else:
                    handle.close()
            except Exception as e:
                print(f"[WARN] Could not process tar {tar_path.name}: {e}")

    def _load_resource(self, rel_path, tar_handle, members_map, is_numpy=False):
        """Unified loader that checks the TAR first, then the local disk."""
        # 1. Try loading from the Archive
        if tar_handle and rel_path in members_map:
            member = members_map[rel_path]
            file_data = tar_handle.extractfile(member).read()
            if is_numpy:
                return np.load(io.BytesIO(file_data))
            return Image.open(io.BytesIO(file_data)).convert("RGB")
        
        # 2. Fallback: Load from local Disk
        disk_path = self.data_root / rel_path
        if not disk_path.exists():
            raise FileNotFoundError(f"Resource {rel_path} not found in TAR or disk folders.")
        
        if is_numpy:
            return np.load(disk_path)
        return Image.open(disk_path).convert("RGB")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text Logic
        label = self.prompt + item["label"]
        caption = item.get("caption", "")

        # Image Logic (Auto-detected Archive or Folder)
        img = self._load_resource(item["image_path"], self.image_tar, self.image_members)
        img = self.image_transform(img)

        # LiDAR Logic (Auto-detected Archive or Folder)
        # Handle cases where lidar is path or coordinates
        if "lidar_path" in item:
            lidar_np = self._load_resource(item["lidar_path"], self.lidar_tar, self.lidar_members, is_numpy=True)
            lidar_data = torch.from_numpy(lidar_np)
        else:
            lidar_data = torch.tensor(item["lidar"])
            
        lidar = self.sparse_to_dense_fn(lidar_data)

        return {
            "label": label,
            "caption": caption,
            "image": img,
            "lidar": lidar
        }

    def __del__(self):
        """Cleanup handles when the dataset is destroyed."""
        if hasattr(self, 'image_tar') and self.image_tar:
            self.image_tar.close()
        if hasattr(self, 'lidar_tar') and self.lidar_tar:
            self.lidar_tar.close()

class Triplet_Scene_Nuscenes(Dataset):
    def __init__(self, jsonl_file, nusc, image_transform, sparse_to_dense_fn):
        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.nusc = nusc  
        self.image_transform = image_transform
        self.sparse_to_dense_fn = sparse_to_dense_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load data
        token = item["sample_token"]
        caption = item.get("caption", "")
        camera_name = item["camera_name"]
        image_path = item["image_path"]
        lidar_path = item["lidar_path"]

        # Image processing
        img = Image.open(image_path).convert("RGB")
        img = self.image_transform(img)

        # Lidar processing
        lidar_scene = load_lidar_bin(lidar_path) # raw lidar
        lidar = zero_pad(lidar_scene, 35000) # [35000, 3]
        _, lidar_scene_non_ground = segment_ground_o3d(lidar_scene) # lidar without ground
        visible_points, _ = lidar2camera_fov(self.nusc, lidar_scene_non_ground, token, camera_name) # camera fov
        visible_points = self.sparse_to_dense_fn(visible_points) # [1024, 3]

        return {
            "token": token,
            "camera_name": camera_name,
            "caption": caption,
            "image": img,
            "lidar": lidar,
            "visible_points": visible_points
        }
