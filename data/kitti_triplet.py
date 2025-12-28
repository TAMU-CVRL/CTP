import os
import json
import tarfile
import io
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
# from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
# from shapely.geometry import Point, Polygon
from utils.data_utils import crop_annotation_kitti
from utils.pc_utils import segment_ground_o3d
def compute_box_corners(x, y, z, h, w, l, ry):
    """
    Compute 8 corners of 3D bounding box (KITTI camera coordinates)
    Args:
        x, y, z: center of box
        h, w, l: box size
        ry: yaw rotation around Y-axis
    Returns:
        corners_3d: (8, 3) array of box corners
    """
    # 定义局部坐标下的8个角点
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

    # 绕Y轴旋转
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    corners_3d = (R @ corners).T
    corners_3d += np.array([x, y, z])
    return corners_3d


def points_in_3d_box(box_corners, points):
    """
    精确判断点是否在3D包围盒内 (NuScenes-style)
    Args:
        box_corners: (8,3) box corner coordinates
        points: (N,3)
    Returns:
        mask: (N,) bool
    """
    # 假设角点顺序如下:
    # 0–3: bottom, 4–7: top
    p1 = box_corners[0]
    p2 = box_corners[1]
    p4 = box_corners[3]
    p5 = box_corners[4]

    # 三个边向量
    i = p2 - p1  # x方向
    j = p4 - p1  # y方向
    k = p5 - p1  # z方向

    v = points - p1.reshape(1, 3)

    iv = np.dot(v, i)
    jv = np.dot(v, j)
    kv = np.dot(v, k)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))

    mask = np.logical_and.reduce((mask_x, mask_y, mask_z))
    return mask


# ===================== 主数据集类 ===================== #
class KITTI_TripletDataset(Dataset):
    def __init__(self, root_dir, split='training', min_points=5):
        """
        Args:
            root_dir: KITTI dataset root (包含 training/testing)
            split: 'training' or 'testing'
            min_points: 最少点云数量阈值
        """
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

        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取点云
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        _, lidar_non_ground = segment_ground_o3d(lidar)

        # 读取标签
        labels = self.read_labels(label_path)

        # 读取标定矩阵
        Tr_velo_to_cam, R0_rect = load_calib(calib_path)

        # 生成 triplets（传入 calib 参数）
        til_triplet, all_bboxes = self.prepare_triplet(image, lidar_non_ground, labels, Tr_velo_to_cam, R0_rect)

        return {
            "til_triplet": til_triplet,
            "all_bboxes": all_bboxes
        }


    def read_labels(self, path):
        """读取 KITTI label_2 文件"""
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
            # 坐标转换
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

    
# class Triplet_Object_KITTI(Dataset):
#     def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, image_tar_path=None, prompt="A "):
#         self.data = []
#         with open(jsonl_file, "r") as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#         self.image_transform = image_transform
#         self.sparse_to_dense_fn = sparse_to_dense_fn
#         self.prompt = prompt
#         self.image_tar_path = image_tar_path

#         if self.image_tar_path is not None:
#             self.tar = tarfile.open(image_tar_path, "r")
#             self.members = {}
#             for m in self.tar.getmembers():
#                 rel_path = m.name
#                 if rel_path.startswith("data/kitti_images/"):
#                     rel_path = rel_path[len("data/kitti_images/"):]
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
#         else:  # Load image from tar instead of disk
#             img_rel_path = item["image_path"]
#             if img_rel_path.startswith("data/kitti_images/"):
#                 img_rel_path = img_rel_path[len("data/kitti_images/"):]

#             if img_rel_path not in self.members:
#                 raise FileNotFoundError(f"{img_rel_path} not found in TAR archive")

#             img_member = self.members[img_rel_path]
#             img_file = self.tar.extractfile(img_member)
#             img = Image.open(io.BytesIO(img_file.read())).convert("RGB")

#         img = self.image_transform(img)

#         # Lidar processing
#         lidar = torch.tensor(item["lidar"], dtype=torch.float32)
#         lidar = self.sparse_to_dense_fn(lidar)

#         return {
#             "label": label,
#             "caption": caption,
#             "image": img,
#             "lidar": lidar
#         }

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
        
        # Text Logic
        label = self.prompt + item["label"]
        caption = item.get("caption", "")

        # Image Logic
        img = self._load_resource(item["image_path"], self.image_tar, self.image_members)
        img = self.image_transform(img)

        # LiDAR Logic
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

# def load_calib(calib_path):
#     """读取 KITTI calib.txt"""
#     calib = {}
#     with open(calib_path, "r") as f:
#         for line in f.readlines():
#             if ":" not in line:
#                 continue
#             key, value = line.split(":", 1)
#             calib[key] = np.array([float(x) for x in value.split()])
#     # 构建矩阵
#     Tr_velo_to_cam = calib.get("Tr_velo_to_cam", np.eye(12)).reshape(3, 4)
#     R0_rect = calib.get("R0_rect", np.eye(9)).reshape(3, 3)
#     return Tr_velo_to_cam, R0_rect
def load_calib(calib_path):
    """Reads KITTI calib.txt and returns transformation matrices."""
    calib = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if ":" not in line: continue
            key, value = line.split(":", 1)
            calib[key] = np.array([float(x) for x in value.split()])
    
    # Construct matrices
    # Tr_velo_to_cam: 3x4 matrix transforming Velodyne to Camera coordinates
    Tr_velo_to_cam = calib.get("Tr_velo_to_cam", np.eye(12)[:12]).reshape(3, 4)
    # R0_rect: 3x3 rectifying rotation matrix
    R0_rect = calib.get("R0_rect", np.eye(9)).reshape(3, 3)
    return Tr_velo_to_cam, R0_rect

def camera_box_to_lidar(x, y, z, h, w, l, ry, Tr_velo_to_cam, R0_rect):
    """
    Converts KITTI 3D bounding box from Camera coordinates to LiDAR coordinates.
    """
    # 1. Box center in Camera homogeneous coordinates
    cam_center = np.array([x, y, z, 1.0])
    
    # 2. Construct 4x4 transformation matrices
    Tr = np.eye(4)
    Tr[:3, :4] = Tr_velo_to_cam
    R0 = np.eye(4)
    R0[:3, :3] = R0_rect
    
    # 3. Compute Camera-to-LiDAR transformation (Inverse of LiDAR-to-Camera)
    T_cam2lidar = np.linalg.inv(R0 @ Tr)
    
    # 4. Transform center coordinates
    lidar_center = (T_cam2lidar @ cam_center)[:3]
    
    # 5. Flip yaw direction (KITTI camera yaw is around Y-axis, LiDAR yaw is around Z-axis)
    yaw_lidar = -ry - np.pi / 2
    
    return lidar_center[0], lidar_center[1], lidar_center[2], h, w, l, yaw_lidar
