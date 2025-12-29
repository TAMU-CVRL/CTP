import os
import json
import tarfile
import io
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class Triplet_Object_Waymo(Dataset):
    def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, prompt=""):
        # 1. Automatic Path Inference
        jsonl_path = Path(jsonl_file).resolve()
        self.data_root = jsonl_path.parent  # Usually dataset/waymo_triplets/
        
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
                    # Standardizing paths: Capture from the start of 'image' or 'lidar'
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
                    print(f"[INFO] Waymo Image TAR detected: {tar_path.name}")
                elif "lidar" in name_lower:
                    self.lidar_tar = handle
                    self.lidar_members = members_map
                    print(f"[INFO] Waymo LiDAR TAR detected: {tar_path.name}")
                else:
                    handle.close()
            except Exception as e:
                print(f"[WARN] Could not process tar {tar_path.name}: {e}")

    def _load_resource(self, rel_path, tar_handle, members_map, is_numpy=False):
        """Unified loader that checks the Archive first, then the local Disk."""
        # Sanitize the relative path to match the member map keys
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
        
        # 2. Fallback to Disk
        disk_path = self.data_root / rel_path
        if not disk_path.exists():
            disk_path = self.data_root / clean_rel
            
        if not disk_path.exists():
            raise FileNotFoundError(f"Resource {rel_path} not found in TAR or disk folders at {self.data_root}")
        
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
            # Direct coordinate lists in JSONL
            lidar_data = torch.tensor(item["lidar"], dtype=torch.float32)
            
        lidar = self.sparse_to_dense_fn(lidar_data)

        return {
            "label": label,
            "caption": caption,
            "image": img,
            "lidar": lidar
        }

    def __del__(self):
        """Clean up handles when dataset is destroyed."""
        if hasattr(self, 'image_tar') and self.image_tar:
            self.image_tar.close()
        if hasattr(self, 'lidar_tar') and self.lidar_tar:
            self.lidar_tar.close()
