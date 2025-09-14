import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import math

from torchvision import transforms
from PIL import Image

def sparse_to_dense(points, target_num = 1024):
    if points.shape[0] >= target_num:
        target_points = farthest_point_sampling(points, npoint=target_num)  # [target_num, 3]
    else:
        scalr = math.ceil(target_num / points.shape[0])
        if scalr > 6:
            factor1 = 6
            factor2 = math.ceil(scalr / 6)
        else:
            factor1 = scalr
            factor2 = 1
        pts = knn_upsample(points, upsample_factor=factor1, k=3)  # [N', 3]
        pts_2 = jitter_upsample(pts, upsample_factor=factor2, jitter_std=0.01)  # [N'', 3]
        target_points = farthest_point_sampling(pts_2, npoint=1024)  # [1024, 3]
    return target_points  # [target_num_points, 3]

# Different upsampling methods
def knn_upsample(points, upsample_factor=2, k=3):
    is_tensor = isinstance(points, torch.Tensor)
    if is_tensor:
        device = points.device
        points = points.cpu().numpy()

    N, D = points.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, indices = nbrs.kneighbors(points)

    new_points = []
    for i in range(N):
        for _ in range(upsample_factor - 1):
            j = np.random.choice(indices[i][1:])  # skip itself
            alpha = np.random.rand()
            interp = alpha * points[i] + (1 - alpha) * points[j]
            new_points.append(interp)

    new_points = np.array(new_points)  # shape: [(upsample_factor - 1) * N, D]
    all_points = np.concatenate([points, new_points], axis=0)

    if is_tensor:
        return torch.tensor(all_points, dtype=torch.float32, device=device)
    else:
        return all_points

def jitter_upsample(points, upsample_factor=2, jitter_std=0.01):
    N, D = points.shape
    num_new = N * (upsample_factor - 1)
    
    idx = torch.randint(0, N, (num_new,), device=points.device)
    new_points = points[idx] + torch.randn((num_new, D), device=points.device) * jitter_std
    all_points = torch.cat([points, new_points], dim=0)

    return all_points  # [N * upsample_factor, D]

# Farthest Point Sampling
def farthest_point_sampling(points, npoint):
    N, C = points.shape
    device = points.device

    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)[0]

    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest].view(1, C)  # [1, C]
        dist = torch.sum((points - centroid) ** 2, dim=1)  # [N]
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=0)[1]

    sampled_points = points[centroids]  # [npoint, C]
    return sampled_points

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