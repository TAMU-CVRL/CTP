import os

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from torchvision import transforms
from PIL import Image

### Pair dataset ###
class TextLiDARPairDataset(Dataset):
    def __init__(self, pairs_list):
        self.pairs_list = pairs_list

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        return self.pairs_list[idx]

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

def save_text_lidar_pair_dataset(SparseCLIP_dataset, save_path):
    flat_pairs_list = []

    for idx in tqdm(range(len(SparseCLIP_dataset)), desc="Flattening & saving TextLiDAR pairs"):
        pairs = SparseCLIP_dataset[idx].get('text_lidar_pair', [])
        for pair in pairs:
            flat_pairs_list.append(pair)

    dataset = TextLiDARPairDataset(flat_pairs_list)
    
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")

    return dataset

def collate_text_lidar_pairs(batch, sparse_to_dense_fn=None):
    labels = ["apointcloud of " + b[0] for b in batch]
    pts_list = [b[1] for b in batch]

    if sparse_to_dense_fn is not None:
        pts_list = [sparse_to_dense_fn(pts) for pts in pts_list]
        pts_list = torch.stack(pts_list, dim=0)

    return labels, pts_list

###  Triplet dataset   ###
def save_triplet_dataset(SparseCLIP_dataset, save_path):
    flat_triplets_list = []

    for idx in tqdm(range(len(SparseCLIP_dataset)), desc="Flattening & saving Text-Image-LiDAR triplets"):
        triplets = SparseCLIP_dataset[idx].get('til_triplet', [])
        for triplet in triplets:
            flat_triplets_list.append(triplet)

    # Save as a torch file
    torch.save(flat_triplets_list, save_path)
    print(f"Triplet list saved to {save_path}")

    return flat_triplets_list

def collate_triplets(batch, sparse_to_dense_fn=None):
    labels = ["a pointcloud of a " + b[0] for b in batch]

    images = [b[1] for b in batch]
    images = [image_transform(img) for img in images]
    images = torch.stack(images, dim=0)

    pts_list = [b[2] for b in batch]

    if sparse_to_dense_fn is not None:
        pts_list = [sparse_to_dense_fn(pts) for pts in pts_list]
        pts_list = torch.stack(pts_list, dim=0)

    return labels, images, pts_list

def load_triplet_dataset(save_path):
    triplets_list = torch.load(save_path)
    return TextLiDARPairDataset(triplets_list)

###  Triplet dataset chunks   ###
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, save_path_prefix):
        # Find all chunk files with the given prefix
        folder = os.path.dirname(save_path_prefix)
        prefix = os.path.basename(save_path_prefix)
        self.chunk_files = sorted([os.path.join(folder, f) 
                                   for f in os.listdir(folder) 
                                   if f.startswith(prefix) and f.endswith(".pt")])
        # Save lengths of each chunk for indexing
        self.chunk_lengths = []
        for f in self.chunk_files:
            data = torch.load(f, map_location='cpu')
            self.chunk_lengths.append(len(data))
        self.cumulative_lengths = [0] + list(torch.cumsum(torch.tensor(self.chunk_lengths), dim=0).numpy())
    
    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find the right chunk
        chunk_idx = max(i for i in range(len(self.cumulative_lengths)) if self.cumulative_lengths[i] <= idx)
        local_idx = idx - self.cumulative_lengths[chunk_idx]
        # Only load the chunk if it's not already in memory
        data = torch.load(self.chunk_files[chunk_idx], map_location='cpu')
        return data[local_idx]

def save_triplet_dataset_chunks(SparseCLIP_dataset, save_path, chunk_size=50000):
    flat_triplets_chunk = []
    chunk_idx = 0

    for idx in tqdm(range(len(SparseCLIP_dataset)), desc="Flattening & saving Text-Image-LiDAR triplets"):
        triplets = SparseCLIP_dataset[idx].get('til_triplet', [])
        for triplet in triplets:
            flat_triplets_chunk.append(triplet)

            # When chunk is full, save to disk
            if len(flat_triplets_chunk) >= chunk_size:
                chunk_file = f"{save_path}_chunk{chunk_idx}.pt"
                torch.save(flat_triplets_chunk, chunk_file)
                print(f"Saved {len(flat_triplets_chunk)} triplets to {chunk_file}")
                flat_triplets_chunk = []
                chunk_idx += 1

    # Save any remaining triplets
    if flat_triplets_chunk:
        chunk_file = f"{save_path}_chunk{chunk_idx}.pt"
        torch.save(flat_triplets_chunk, chunk_file)
        print(f"Saved {len(flat_triplets_chunk)} triplets to {chunk_file}")

    print("All triplets saved successfully.")

def resize_with_aspect_ratio(img, size=256):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    new_img = Image.new("RGB", (size, size))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    return new_img

image_transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_aspect_ratio(img, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP mean
        std=(0.26862954, 0.26130258, 0.27577711)   # CLIP std
    )
])
