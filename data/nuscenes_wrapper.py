import numpy as np

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box

import torch
from torch.utils.data import Dataset

from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.utils.geometry_utils import view_points
import cv2
import numpy as np
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
            tp_pair, _ = self.prepare_text_lidar_pair(sample, last_lidar)
        sample['text_lidar_pair'] = tp_pair
        return sample

    def box2image(self, sample, ann_token, cam_channel="CAM_FRONT"):
        cam_token = sample['data'][cam_channel]
        cam_data = self.nusc.get('sample_data', cam_token)
        cam_path, boxes, cam_intrinsic = self.nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])

        im = Image.open(cam_path).convert("RGB")
        
        box = boxes[0]
        
        corners = view_points(box.corners(), np.array(cam_intrinsic), normalize=True)[:2, :]
        x_min, y_min = corners.min(axis=1)
        x_max, y_max = corners.max(axis=1)

        crop = im.crop((x_min, y_min, x_max, y_max))
        return crop
    
    def prepare_text_lidar_pair(self, sample, pc_ego):
        """
        sample: a sample from NuScenes
        pc_ego: point cloud in ego coordinate, shape [N, 3] or [N, 4] (x, y, z [, intensity])
        """
        sample_record = self.nusc.get('sample', sample['token'][-1])
        lidar_token = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', lidar_token['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', lidar_token['ego_pose_token'])
        tp_pair = []
        all_bboxes = []

        for ann_token in sample_record['anns']:
            obj = self.nusc.get('sample_annotation', ann_token)
            full_label = obj['category_name']
            parts = full_label.split('.')
            # Use the second part as the label if it exists, otherwise use the last part
            if len(parts) >= 2:
                label = parts[1]
            else:
                label = parts[-1]
            # Get 3D bounding box
            box = Box(obj['translation'], obj['size'], Quaternion(obj['rotation']))

            # Step 1: global -> ego
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            # Step 2: ego -> lidar
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            yaw = box.orientation.yaw_pitch_roll[0]
            all_bboxes.append({
                'sample_token': sample_record['token'],
                'instance_token': obj['instance_token'],
                'bbox': [*box.center, *box.wlh, yaw],
                'category': obj['category_name']
            })
            
            # Extract points within the bounding box
            mask = points_in_box(box, pc_ego[:, :3].T)
            points_in_instance = pc_ego[mask]

            # Skip if too few points
            # if len(points_in_instance) < 20:
            #     continue
            # else:
            #     pts = knn_upsample(points_in_instance, upsample_factor=6, k=3)  # [N', 3]
            #     pts_2 = jitter_upsample(pts, upsample_factor=10, jitter_std=0.01)  # [N'', 3]
            #     pts_3 = farthest_point_sampling(pts_2, npoint=1024)  # [1024, 3]
            # tp_pair.append((label, pts_3)) # [(label, pts)]]
            tp_pair.append((label, points_in_instance)) # [(label, pts)]]

        return tp_pair, all_bboxes

class TextLiDARPairDataset(Dataset):
    def __init__(self, pairs_list):
        self.pairs_list = pairs_list

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        return self.pairs_list[idx]

# def collate_text_lidar_pairs(batch):
#     labels = [b[0] for b in batch]   # string label
#     pts = [b[1] for b in batch]      # tensor of different lengths
#     return labels, pts

def collate_text_lidar_pairs(batch, sparse_to_dense_fn=None):
    labels = ["a pointcloud of " + b[0] for b in batch]
    pts_list = [b[1] for b in batch]

    if sparse_to_dense_fn is not None:
        pts_list = [sparse_to_dense_fn(pts) for pts in pts_list]
        pts_list = torch.stack(pts_list, dim=0)

    return labels, pts_list

def save_text_lidar_pair_dataset(SparseCLIP_dataset, save_path='data/text_lidar_pair_dataset.pt'):
    flat_pairs_list = []

    for idx in tqdm(range(len(SparseCLIP_dataset)), desc="Flattening & saving TextLiDAR pairs"):
        pairs = SparseCLIP_dataset[idx].get('text_lidar_pair', [])
        for pair in pairs:
            flat_pairs_list.append(pair)

    dataset = TextLiDARPairDataset(flat_pairs_list)
    
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")

    return dataset

def filter_small_pointclouds(dataset, min_points=20):
    filtered_pairs = []
    total_samples = len(dataset)
    filtered_out_count = 0

    for label, pts in tqdm(dataset, desc="Filtering small pointclouds"):
        if pts.shape[0] >= min_points:
            filtered_pairs.append((label, pts))
        else:
            filtered_out_count += 1

    remaining_samples = len(filtered_pairs)
    print(f"Original samples: {total_samples}")
    print(f"Filtered out (less than {min_points} points): {filtered_out_count}")
    print(f"Remaining samples: {remaining_samples}")

    filtered_dataset = TextLiDARPairDataset(filtered_pairs)

    return filtered_dataset
