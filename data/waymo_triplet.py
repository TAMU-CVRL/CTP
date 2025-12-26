import json, tarfile, io, torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Triplet_Object_Waymo(Dataset):
    def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn, image_tar_path=None, lidar_tar_path=None, prompt="A "):
        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_transform = image_transform
        self.sparse_to_dense_fn = sparse_to_dense_fn
        self.prompt = prompt
        self.image_tar_path = image_tar_path

        # Optional TAR loading
        self.image_tar_path = image_tar_path
        if self.image_tar_path is not None:
            self.image_tar = tarfile.open(self.image_tar_path, "r")
            self.image_members = {}
            for m in self.image_tar.getmembers():
                rel_path = m.name
                if rel_path.startswith("data/waymo_images/"):
                    rel_path = rel_path[len("data/waymo_images/"):]
                self.image_members[rel_path] = m
            print(f"[INFO] Images loaded from TAR archive: {self.image_tar_path}")
        else:
            print(f"[INFO] Images will be loaded from disk directly.")
        # Optional TAR loading
        self.lidar_tar_path = lidar_tar_path
        if self.lidar_tar_path is not None:
            self.lidar_tar = tarfile.open(self.lidar_tar_path, "r")
            self.lidar_members = {}
            for m in self.lidar_tar.getmembers():
                rel_path = m.name
                if rel_path.startswith("data/waymo_lidars/"):
                    rel_path = rel_path[len("data/waymo_lidars/"):]
                self.lidar_members[rel_path] = m
            print(f"[INFO] LiDARs loaded from TAR archive: {self.lidar_tar_path}")
        else:
            print(f"[INFO] LiDARs will be loaded from disk directly.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Text processing
        label = self.prompt + item["label"]  # e.g. "A Vehicle"
        caption = item.get("caption", "")
        
        # Image processing
        if self.image_tar_path is None:
            # Directly from disk
            img = Image.open(item["image_path"]).convert("RGB")
        else:
            # From TAR archive
            img_rel_path = item["image_path"]
            if img_rel_path.startswith("data/waymo_images/"):
                img_rel_path = img_rel_path[len("data/waymo_images/"):]
            if img_rel_path not in self.image_members:
                raise FileNotFoundError(f"{img_rel_path} not found in TAR archive")

            img_member = self.image_members[img_rel_path]
            img_file = self.tar.extractfile(img_member)
            img = Image.open(io.BytesIO(img_file.read())).convert("RGB")

        img = self.image_transform(img)

        # Lidar processing
        if self.lidar_tar_path is None:
            lidar = np.load(item["lidar_path"])
        else:
            lidar_rel_path = item["lidar_path"]
            if lidar_rel_path.startswith("data/waymo_lidars/"):
                lidar_rel_path = lidar_rel_path[len("data/waymo_lidars/"):]
            if lidar_rel_path not in self.lidar_members:
                raise FileNotFoundError(f"{lidar_rel_path} not found in TAR archive")
            lidar_member = self.lidar_members[lidar_rel_path]
            lidar_file = self.lidar_tar.extractfile(lidar_member)
            lidar = np.load(io.BytesIO(lidar_file.read()))

        lidar = torch.tensor(lidar, dtype=torch.float32)
        lidar = self.sparse_to_dense_fn(lidar)

        return {
            "label": label,
            "caption": caption,
            "image": img,
            "lidar": lidar,
        }
