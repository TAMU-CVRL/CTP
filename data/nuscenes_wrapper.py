import numpy as np
import os
from pathlib import Path

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box

import torch
from torch.utils.data import Dataset

from pyquaternion import Quaternion
from tqdm import tqdm

import numpy as np

from utils.box2image import crop_annotation
import json
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import InterpolationMode
from PIL import Image

class SparseCLIP_Dataset(Dataset):
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
            all_bboxes.append({
                'sample_token': ann_record['token'],
                'instance_token': ann_record['instance_token'],
                'bbox': [*box.center, *box.wlh, yaw],
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
            til_triplet.append((label, cropped_image, points_in_instance))
        return til_triplet, all_bboxes

class TripletJsonlDataset(Dataset):
    def __init__(self, jsonl_file, image_transform, sparse_to_dense_fn):
        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.image_transform = image_transform
        self.sparse_to_dense_fn = sparse_to_dense_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text processing
        label = "a pointcloud of a " + item["label"]

        # Image processing
        img = Image.open(item["image_path"]).convert("RGB")
        img = self.image_transform(img)

        # Lidar processing
        lidar = torch.tensor(item["lidar"])
        lidar = self.sparse_to_dense_fn(lidar)

        return label, img, lidar

def save_triplet_dataset_jsonl(SparseCLIP_dataset, save_jsonl_path, image_dir, split, image_format='png'):
    os.makedirs(image_dir, exist_ok=True)

    # Open JSONL with line buffering for safety
    with open(save_jsonl_path, "w", buffering=1) as f_out:  # line buffering
        for idx in tqdm(range(len(SparseCLIP_dataset)), desc="Processing triplets"):
            triplets = SparseCLIP_dataset[idx].get('til_triplet', [])
            for i, triplet in enumerate(triplets):
                label, img_pil, lidar = triplet

                # Convert LiDAR tensor to list
                lidar_list = lidar.tolist()

                # Save image to disk
                img_filename = f"{split}_{idx}_{i}_{label}.{image_format}"
                img_path = os.path.join(image_dir, img_filename)
                img_pil.save(img_path)

                # Save JSON line
                json_obj = {
                    "label": label,
                    "image_path": img_path,
                    "lidar": lidar_list
                }
                f_out.write(json.dumps(json_obj) + "\n")
                f_out.flush()  # flush after each line

    print(f"Triplets saved as JSONL to {save_jsonl_path} with images in {image_dir}")
