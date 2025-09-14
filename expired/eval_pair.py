import os
import yaml

from models.sllmclip import sclip_pair
from models.pointnet2 import pointnet2_encoder
from functions import TextLiDARPairDataset, collate_text_lidar_pairs, filter_small_pointclouds
from utils.processing import sparse_to_dense

from pathlib import Path
import torch

import clip
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from transformers import get_cosine_schedule_with_warmup
from collections import defaultdict

def evaluate_classification_accuracy(model, dataloader, device):
    
    model.eval()  # Set model to evaluation mode

    # List of class names in your dataset
    base_classes = [
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
    all_classes = [f"a pointcloud of {cls}" for cls in base_classes]

    # Dictionaries to keep track of correct predictions and total counts per class
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    # Tokenize all class names once and send to device (for text encoder)
    class_tokens = clip.tokenize(all_classes, truncate=True).to(device)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for sample in tqdm(dataloader, desc="Evaluating Classification Accuracy"):

            labels = sample[0]   # string label
            points = sample[1].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
            text_ids = clip.tokenize(labels, truncate=True).to(device)

            # Forward pass all valid patches at once (better than loop)
            logits_per_lidar, _ = model(points, class_tokens)  # [N, C]
            preds = logits_per_lidar.argmax(dim=1).cpu()

            # Update counts
            for pred_idx, gt_class in zip(preds, labels):
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

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
###############################################################################
# Main function
if __name__ == "__main__":
    cfg = load_config("configs/sclip.yaml")
    device = cfg["Train"]["device"] if torch.cuda.is_available() else "cpu"

    ### Load models ###
    # Clip text encoder
    clip_model, clip_preprocess = clip.load(cfg["Model"]["clip_model"], jit=False)
    clip_model.to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    text_encoder = clip_model.encode_text
    # PointNet++ encoder
    lidar_encoder = pointnet2_encoder.PointNet2Encoder() # [B, C, N] -> [B, C']
    lidar_encoder.to(device)
    # model
    model = sclip_pair(text_encoder=text_encoder, lidar_encoder=lidar_encoder)
    model.to(device)
    print(f"Model parameters: {model.param_count}")
    print(f"Lidar encoder parameters: {lidar_encoder.param_count}")

    ### Load dataset ###
    batch_size = cfg["Train"]["batch_size"]
    min_points = cfg["Dataset"]["min_points"]

    eval_dataset = torch.load(cfg["Dataset"]["eval_data_path"])
    filtered_eval_dataset = filter_small_pointclouds(eval_dataset, min_points=min_points)
    eval_dataloader = torch.utils.data.DataLoader(
        filtered_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_text_lidar_pairs(batch, sparse_to_dense_fn=sparse_to_dense)
    )
    
    ### Evaluate the model ###
    checkpoint_path = "/home/ximeng/Documents/SparseCLIP/checkpoints/checkpoint_epoch5.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        class_acc, overall_acc = evaluate_classification_accuracy(model, eval_dataloader, device)
    output_path = "evaluation_results.txt"
    with open(output_path, "w") as f:
        f.write("Per-class accuracies:\n")
        for cls, acc in class_acc.items():
            line = f"{cls}: {acc*100:.2f}%\n"
            print(line, end="")
            f.write(line)
        overall_line = f"Overall Accuracy: {overall_acc*100:.2f}%\n"
        print(overall_line, end="")
        f.write(overall_line)
        