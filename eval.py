import torch
from tqdm import tqdm
import clip
import torch.nn.functional as F

from models.sllmclip import sllm_clip
from models.pointnet2 import pointnet2_encoder
from data.nuscenes_data import NuscenesData
from SparseCLIP.data.nuscenes_wrapper import SLLMCLIP_classification

from nuscenes.nuscenes import NuScenes
from pathlib import Path
from collections import defaultdict
import open3d as o3d
import random
import numpy as np

def visualize_point_cloud(points, point_size=1.0):
    """
    Visualize a point cloud using Open3D.

    Args:
        points: numpy array of shape [N, 3], or torch tensor.
        point_size: float, size of the points in visualization.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy().squeeze().astype(np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be of shape [N, 3]")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    length = points.shape[0]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # gray

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud, Length: {length}", width=800, height=600)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0, 0, 0])  # black background

    vis.run()
    vis.destroy_window()
    
def evaluate_classification_accuracy_per_class(model, dataloader, device):
    
    model.eval()  # Set model to evaluation mode

    # List of class names in your dataset
    all_classes = [
        'car',
        'truck',
        'bus',
        'motorcycle',
        'bicycle',
        'trailer',
        'construction',
        'pedestrian',
        'trafficcone',
        'barrier'
    ]
    # Create a mapping from class name to index for convenience
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # Dictionaries to keep track of correct predictions and total counts per class
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    # Tokenize all class names once and send to device (for text encoder)
    class_tokens = clip.tokenize(all_classes, truncate=True).to(device)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for examples in tqdm(dataloader, desc="Evaluating Classification Accuracy"):
            if examples is None:
                continue
            # Extract point cloud data, mask, and labels from batch
            pts = examples['text_lidar_pts'].to(device)        # Shape: [B, max_len, 1024, 3]
            mask = examples['text_lidar_mask'].to(device)      # Shape: [B, max_len]
            labels = examples['text_lidar_labels']             # List of lists of strings: [B][max_len]

            B, max_len, _, _ = pts.shape

            # Rearrange points dimensions to match model input [B, max_len, 3, 1024]
            pts = pts.permute(0, 1, 3, 2)

            # Select only valid points where mask is True, then flatten to [N, 3, 1024]
            valid_pts = pts[mask].reshape(-1, 3, 1024)

            # Flatten ground truth labels corresponding to valid points
            gt_text_labels = []
            for b in range(B):
                for j in range(max_len):
                    if mask[b, j]:
                        gt_text_labels.append(labels[b][j])

    #         # Loop over each valid point cloud patch
    #         for i in range(valid_pts.size(0)):
    #             lidar_feat = valid_pts[i].unsqueeze(0)  # Add batch dim: [1, 3, 1024]

    #             # Forward pass: get logits over classes from model
    #             logits_per_lidar, _ = model(lidar_feat, class_tokens)  # Shape: [1, C]

    #             # Predict class index with highest logit
    #             pred_idx = logits_per_lidar.argmax(dim=1).item()

    #             pred_class = all_classes[pred_idx]
    #             gt_class = gt_text_labels[i]

    #             # Update counts
    #             total_counts[gt_class] += 1
    #             if pred_class == gt_class:
    #                 correct_counts[gt_class] += 1

    # # Compute accuracy per class, None if no samples for that class
    # class_accuracies = {}
    # for cls in all_classes:
    #     if total_counts[cls] > 0:
    #         class_accuracies[cls] = correct_counts[cls] / total_counts[cls]
    #     else:
    #         class_accuracies[cls] = None

    # return class_accuracies

            # Forward pass all valid patches at once (better than loop)
            logits_per_lidar, _ = model(valid_pts, class_tokens)  # [N, C]
            preds = logits_per_lidar.argmax(dim=1).cpu()

            # Update counts
            for pred_idx, gt_class in zip(preds, gt_text_labels):
                pred_class = all_classes[pred_idx]
                total_counts[gt_class] += 1
                if pred_class == gt_class:
                    correct_counts[gt_class] += 1

    # Compute accuracy per class
    class_accuracies = {
        cls: (correct_counts[cls] / total_counts[cls] if total_counts[cls] > 0 else None)
        for cls in all_classes
    }

    # Compute overall accuracy
    total_correct = sum(correct_counts.values())
    total_samples = sum(total_counts.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

    return class_accuracies, overall_acc
    
def evaluate_recall(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for examples in tqdm(dataloader, desc="Evaluating Recall@1"):
            pts = examples['text_lidar_pts'].to(device)        # [B, max_len, 1024, 3]
            mask = examples['text_lidar_mask'].to(device)      # [B, max_len]
            labels = examples['text_lidar_labels']             # list[list[str]]

            B, max_len, _, _ = pts.shape
            pts = pts.permute(0, 1, 3, 2)   # [B, max_len, 3, 1024]
            valid_pts = pts[mask].reshape(-1, 3, 1024)

            valid_texts = []
            for b in range(B):
                for j in range(max_len):
                    if mask[b, j]:
                        valid_texts.append(labels[b][j])

            text_ids = clip.tokenize(valid_texts, truncate=True).to(device)

            logits_per_lidar, logits_per_text = model(valid_pts, text_ids)  # [N, N]
            similarity = logits_per_lidar  # cosine similarity matrix

            preds = similarity.argmax(dim=1)  # [N]
            labels = torch.arange(similarity.size(0)).to(device)  # assume diagonal is ground truth

            correct += (preds == labels).sum().item()
            total += similarity.size(0)

    recall_at_1 = correct / total if total > 0 else 0
    return recall_at_1

def evaluate_classification_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    all_classes = [
        'car',
        'truck',
        'bus',
        'motorcycle',
        'bicycle',
        'trailer',
        'construction',
        'pedestrian',
        'trafficcone',
        'barrier'
    ]

    class_tokens = clip.tokenize(all_classes, truncate=True).to(device)  # [C, seq_len]

    with torch.no_grad():
        for examples in tqdm(dataloader, desc="Evaluating Classification Accuracy"):
            if examples is None:
                continue            
            pts = examples['text_lidar_pts'].to(device)        # [B, max_len, 1024, 3]
            mask = examples['text_lidar_mask'].to(device)      # [B, max_len]
            labels = examples['text_lidar_labels']             # list[list[str]]

            B, max_len, _, _ = pts.shape
            pts = pts.permute(0, 1, 3, 2)                      # [B, max_len, 3, 1024]
            valid_pts = pts[mask].reshape(-1, 3, 1024)

            # Flatten GT labels for valid pts
            gt_text_labels = []
            for b in range(B):
                for j in range(max_len):
                    if mask[b, j]:
                        gt_text_labels.append(labels[b][j])    # e.g., 'car'

            # For each valid point, compute similarity with all class tokens
            for i in range(valid_pts.size(0)):
                lidar_feat = valid_pts[i].unsqueeze(0)         # [1, 3, 1024]
                logits_per_lidar, _ = model(lidar_feat, class_tokens)  # [1, C]
                pred_idx = logits_per_lidar.argmax(dim=1).item()
                pred_class = all_classes[pred_idx]
                gt_class = gt_text_labels[i]

                if pred_class == gt_class:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

def predict_single_pointcloud(model, pointcloud, device):
    model.eval()

    all_classes = [
        'car',
        'truck',
        'bus',
        'motorcycle',
        'bicycle',
        'trailer',
        'construction',
        'pedestrian',
        'trafficcone',
        'barrier'
    ]

    class_tokens = clip.tokenize(all_classes, truncate=True).to(device)  # [C, seq_len]

    with torch.no_grad():
        pointcloud = pointcloud.to(device)        # [3, 1024]
        pointcloud = pointcloud.transpose(0, 1)   # [1024, 3]
        pointcloud = pointcloud.unsqueeze(0)      # [1, 1024, 3]

        logits_per_lidar, _ = model(pointcloud, class_tokens)  # [1, C]
        pred_idx = logits_per_lidar.argmax(dim=1).item()
        pred_class = all_classes[pred_idx]

    return pred_class

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
### Load models ###
# Clip text encoder
clip_model, clip_preprocess = clip.load("ViT-B/32", jit=False)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
text_encoder = clip_model.encode_text
# PointNet++ encoder
lidar_encoder = pointnet2_encoder.PointNet2Encoder() # [B, C, N] -> [B, C']
lidar_encoder.to(device)
# model
model = sllm_clip(text_encoder=text_encoder, lidar_encoder=lidar_encoder)
print(f"Model parameters: {model.param_count}")
print(f"Lidar encoder parameters: {lidar_encoder.param_count}")

checkpoint_path = "checkpoints_class/checkpoint_epoch4.pt"
print(f"Loading model from {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)

def pad_text_lidar_pair(batch_pairs, max_len=None):
    B = len(batch_pairs)
    if max_len is None:
        max_len = max(len(pairs) for pairs in batch_pairs)

    padded_pts = torch.zeros(B, max_len, 1024, 3, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    padded_labels = [[''] * max_len for _ in range(B)]
    flat_labels = []

    for i, pairs in enumerate(batch_pairs):
        if len(pairs) == 0:
            continue
        for j, (label, pts) in enumerate(pairs):
            if j >= max_len:
                break
            if pts.shape != (1024, 3):
                raise ValueError(f"Expected pts shape (1024, 3), got {pts.shape}")
            padded_pts[i, j] = pts
            padded_labels[i][j] = label
            mask[i, j] = True
            flat_labels.append(label)

    return padded_pts, padded_labels, mask, flat_labels

def custom_collate_fn(batch):
    collated = {}
    keys = batch[0].keys()
    for k in keys:
        values = [b[k] for b in batch]
        if k == 'text_lidar_pair':
            values = [v for v in values if len(v) > 0]
            if len(values) == 0:
                return None 
            
            pts, labels, mask, flat_labels = pad_text_lidar_pair(values)
            collated['text_lidar_pts'] = pts
            collated['text_lidar_labels'] = labels
            collated['text_lidar_mask'] = mask
            collated['text_lidar_flat_labels'] = flat_labels
        else:
            try:
                collated[k] = torch.utils.data._utils.collate.default_collate(values)
            except Exception:
                collated[k] = values
    return collated
### Load dataset ###
data_path = Path("/home/ximeng/Dataset/nuscenes_full_v1_0/")
nusc = NuScenes(version='v1.0-trainval', dataroot=data_path)
dataset = NuscenesData(nusc, 0, 2, 0)
CLIP_test_dataset = SLLMCLIP_classification(dataset)
CLIP_test_dataloader  = torch.utils.data.DataLoader(CLIP_test_dataset, batch_size=12, shuffle=False, collate_fn=custom_collate_fn)

class_accuracies, overall_acc = evaluate_classification_accuracy_per_class(model, CLIP_test_dataloader, device)

with open("class_accuracies.txt", "w") as f:
    for cls, acc in class_accuracies.items():
        if acc is None:
            line = f"{cls}: No samples in dataset\n"
        else:
            line = f"{cls}: {acc*100:.2f}% accuracy\n"
        print(line.strip())
        f.write(line)

    overall_line = f"\nOverall Accuracy: {overall_acc*100:.2f}%\n"
    print(overall_line.strip())
    f.write(overall_line)
