import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
from nuscenes.utils.geometry_utils import view_points

# def save_triplet_dataset_jsonl(dataset, save_jsonl_path, image_dir, split, image_format='png'):
#     os.makedirs(image_dir, exist_ok=True)

#     # Open JSONL with line buffering for safety
#     with open(save_jsonl_path, "w", buffering=1) as f_out:  # line buffering
#         for idx in tqdm(range(len(dataset)), desc="Processing triplets"):
#             triplets = dataset[idx].get('til_triplet', [])
#             if not triplets:
#                 continue
#             for i, triplet in enumerate(triplets):
#                 label, img_pil, lidar, bbox = triplet

#                 # Convert LiDAR tensor to list
#                 lidar_list = lidar.tolist()

#                 # Save image to disk
#                 img_filename = f"{split}_{idx}_{i}_{label}.{image_format}"
#                 img_path = os.path.join(image_dir, img_filename)
#                 img_pil.save(img_path)

#                 # Save JSON line
#                 json_obj = {
#                     "label": label,
#                     "image_path": img_path,
#                     "lidar": lidar_list,
#                     "bbox": bbox
#                 }
#                 f_out.write(json.dumps(json_obj) + "\n")
#                 f_out.flush()  # flush after each line

#     print(f"[DONE] Triplets saved to {save_jsonl_path}, images in {image_dir}")

# def save_triplet_dataset_jsonl(dataset, save_jsonl_path, split, rel_image_dir, rel_lidar_dir, image_format='png', lidar_format='npy'):
#     jsonl_abs_path = Path(save_jsonl_path).resolve()
#     base_root = jsonl_abs_path.parent

#     abs_image_dir = base_root / rel_image_dir
#     abs_lidar_dir = base_root / rel_lidar_dir

#     abs_image_dir.mkdir(parents=True, exist_ok=True)
#     abs_lidar_dir.mkdir(parents=True, exist_ok=True)
#     # Open JSONL with line buffering for safety
#     with open(save_jsonl_path, "w", buffering=1) as f_out:  # line buffering
#         for idx in tqdm(range(len(dataset)), desc="Processing triplets"):
#             triplets = dataset[idx].get('til_triplet', [])
#             if not triplets:
#                 continue
#             for i, triplet in enumerate(triplets):
#                 label, img_pil, lidar, bbox = triplet

#                 # Save image to disk
#                 img_filename = f"{split}_{idx}_{i}_{label}.{image_format}"
#                 rel_img_path = os.path.join(rel_image_dir, img_filename)
#                 abs_img_path = os.path.join(abs_image_dir, img_filename)
#                 img_pil.save(abs_img_path)

#                 # Save LiDAR to disk
#                 lidar_filename = f"{split}_{idx}_{i}_{label}.{lidar_format}"
#                 rel_lidar_path = os.path.join(rel_lidar_dir, lidar_filename)
#                 abs_lidar_path = os.path.join(abs_lidar_dir, lidar_filename)
#                 if lidar_format == 'npy':
#                     np.save(abs_lidar_path, lidar)
#                 elif lidar_format == 'txt':
#                     np.savetxt(abs_lidar_path, lidar, fmt="%.4f")
#                 else:
#                     raise ValueError(f"Unsupported lidar_format: {lidar_format}")
                
#                 # Save JSON line
#                 json_obj = {
#                     "label": label,
#                     "image_path": rel_img_path,
#                     "lidar_path": rel_lidar_path,
#                     "bbox": bbox
#                 }
#                 f_out.write(json.dumps(json_obj) + "\n")
#                 f_out.flush()  # flush after each line

#     print(f"[DONE] Triplets saved to {save_jsonl_path}, images in {abs_image_dir}, LiDAR in {abs_lidar_dir}")

def save_triplet_dataset_jsonl(dataset, save_jsonl_path, split, rel_image_dir, rel_lidar_dir, image_format='png', lidar_format='npy'):
    # 1. Anchor the absolute path relative to where the JSONL is stored
    jsonl_abs_path = Path(save_jsonl_path).resolve()
    base_root = jsonl_abs_path.parent

    # 2. Define Absolute Paths for Disk I/O
    abs_image_dir = base_root / rel_image_dir
    abs_lidar_dir = base_root / rel_lidar_dir

    # 3. Create directories physically
    abs_image_dir.mkdir(parents=True, exist_ok=True)
    abs_lidar_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving images to: {abs_image_dir}")
    print(f"[INFO] Saving LiDAR to: {abs_lidar_dir}")

    # Open JSONL with line buffering
    with open(save_jsonl_path, "w", buffering=1) as f_out:
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split} triplets"):
            triplets = dataset[idx].get('til_triplet', [])
            if not triplets:
                continue

            for i, triplet in enumerate(triplets):
                label, img_pil, lidar, bbox = triplet

                # Common Filename Logic
                filename_base = f"{split}_{idx}_{i}_{label}"
                img_filename = f"{filename_base}.{image_format}"
                lidar_filename = f"{filename_base}.{lidar_format}"

                # --- 1. Disk Saving (Use Absolute Paths) ---
                abs_img_path = abs_image_dir / img_filename
                abs_lidar_path = abs_lidar_dir / lidar_filename
                
                # Save Image
                img_pil.save(str(abs_img_path))

                # Save LiDAR
                if lidar_format == 'npy':
                    np.save(str(abs_lidar_path), lidar)
                elif lidar_format == 'txt':
                    np.savetxt(str(abs_lidar_path), lidar, fmt="%.4f")
                else:
                    raise ValueError(f"Unsupported lidar_format: {lidar_format}")

                # --- 2. JSON Record (Use Relative Paths) ---
                # We use os.path.join or Path / to combine the relative dir prefix with filename
                json_obj = {
                    "label": label,
                    "image_path": str(Path(rel_image_dir) / img_filename),
                    "lidar_path": str(Path(rel_lidar_dir) / lidar_filename),
                    "bbox": bbox
                }
                
                f_out.write(json.dumps(json_obj) + "\n")
                # f_out.flush() # Not strictly necessary if buffering=1 is set

    print(f"[DONE] Triplets saved to {save_jsonl_path}")
    
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py, render_annotation
def crop_annotation_nusc(nusc, ann_token, sample_record, margin=5, min_ratio=0.8):
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=1, selected_anntokens=[ann_token])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    if len(boxes) == 0:
        return None  # skip if not visible

    cam_token = sample_record['data'][cam]

    # Plot CAMERA view.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
    im = Image.open(data_path)

    # Crop the box from the image
    box = boxes[0]
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    x_min, y_min = corners.min(axis=1)
    x_max, y_max = corners.max(axis=1)
    
    # Calculate the area inside the image, if too small, skip this box
    x_min_clip = max(int(x_min), 0)
    y_min_clip = max(int(y_min), 0)
    x_max_clip = min(int(x_max), im.width)
    y_max_clip = min(int(y_max), im.height)

    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    if ratio_inside < min_ratio:
        return None

    # Add margin and ensure within image bounds
    x_min_final = max(int(x_min) - margin, 0)
    y_min_final = max(int(y_min) - margin, 0)
    x_max_final = min(int(x_max) + margin, im.width)
    y_max_final = min(int(y_max) + margin, im.height)

    cropped_im = im.crop((x_min_final, y_min_final, x_max_final, y_max_final)) 
    
    return cropped_im

def crop_annotation_kitti(image: Image.Image, bbox_2d, margin: int = 5, min_ratio: float = 0.8):
    width, height = image.size
    x_min, y_min, x_max, y_max = bbox_2d

    # calculate box area
    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)

    # clip to image boundaries
    x_min_clip = max(int(x_min), 0)
    y_min_clip = max(int(y_min), 0)
    x_max_clip = min(int(x_max), width)
    y_max_clip = min(int(y_max), height)

    # calculate clipped area
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    # skip if too small
    if ratio_inside < min_ratio or area_clipped <= 0:
        return None

    # Add margin and ensure within image bounds
    x_min_final = max(int(x_min_clip) - margin, 0)
    y_min_final = max(int(y_min_clip) - margin, 0)
    x_max_final = min(int(x_max_clip) + margin, width)
    y_max_final = min(int(y_max_clip) + margin, height)

    # crop
    cropped_im = image.crop((x_min_final, y_min_final, x_max_final, y_max_final))

    return cropped_im
