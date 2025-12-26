import numpy as np
import io
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.pc_utils import lidar2camera_fov, segment_ground_o3d, zero_pad
import tarfile

def load_lidar_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # or :4 if 5 columns
    return torch.tensor(points[:, :3], dtype=torch.float32)  # Keep only xyz

class Triplet_Scene(Dataset):
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

class Triplet_Object(Dataset):
    def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, image_tar_path = None, prompt="A "):
        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_transform = image_transform
        self.sparse_to_dense_fn = sparse_to_dense_fn
        self.prompt = prompt
        self.image_tar_path = image_tar_path
        if self.image_tar_path is not None:
            # Open tar archive once
            self.tar = tarfile.open(image_tar_path, "r")
            self.members = {}
            for m in self.tar.getmembers():
                rel_path = m.name
                if rel_path.startswith("data/nuscenes_images/"):
                    rel_path = rel_path[len("data/nuscenes_images/"):]
                self.members[rel_path] = m
            print(f"[INFO] Images will be loaded from TAR archive: {image_tar_path}")
        else:
            print(f"[INFO] Images will be loaded from disk directly.")    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text processing
        label = self.prompt + item["label"]
        caption = item.get("caption", "")

        # Image processing
        if self.image_tar_path is None: # Load image from disk directlyimage
            img = Image.open(item["image_path"]).convert("RGB")
        else: # Load image from tar instead of disk
            img_rel_path = item["image_path"]
            if img_rel_path.startswith("data/nuscenes_images/"):
                img_rel_path = img_rel_path[len("data/nuscenes_images/"):]
            
            if img_rel_path not in self.members:
                raise FileNotFoundError(f"{img_rel_path} not found in tar archive")

            img_member = self.members[img_rel_path]
            img_file = self.tar.extractfile(img_member)
            img = Image.open(io.BytesIO(img_file.read())).convert("RGB")

        img = self.image_transform(img)

        # Lidar processing
        lidar = torch.tensor(item["lidar"])
        lidar = self.sparse_to_dense_fn(lidar)

        return {
            "label": label,
            "caption": caption,
            "image": img,
            "lidar": lidar
        }
    