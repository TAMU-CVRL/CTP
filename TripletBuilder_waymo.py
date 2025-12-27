# TripletBuilder_waymo.py
# ----------------------------------------
# Extract text-lidar-image triplets from Waymo Open Perception Dataset (v2.0.1)
# and save as JSONL file with cropped images and LiDAR point clouds.
# ----------------------------------------

import os
import glob
import json
from io import BytesIO
import argparse
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from waymo_open_dataset.v2 import (
    LiDARCalibrationComponent,   # LiDAR intrinsic and extrinsic calibration parameters
    LiDARBoxComponent,           # 3D bounding boxes defined in LiDAR (vehicle) coordinates
    LiDARComponent,              # Raw LiDAR range image and point cloud data
    CameraImageComponent,        # Camera RGB frames with metadata
    ProjectedLiDARBoxComponent   # LiDAR 3D boxes projected into the camera image plane (2D boxes)
)
from waymo_open_dataset.v2.perception.utils import lidar_utils

# WOD full label mapping
WAYMO_LABEL_MAP = {
    0: "Unknown",
    1: "Vehicle",
    2: "Pedestrian",
    3: "Sign",
    4: "Cyclist",
}

### Utility Functions ###
def segment_ground_o3d(points, dist_thresh=0.15, ransac_n=3, num_iterations=1000):
    if points.shape[1] > 3:
        points = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    ground_points = points[inliers]
    non_ground_points = np.delete(points, inliers, axis=0)
    # print(f"[INFO] Ground points: {len(ground_points)}, Non-ground points: {len(non_ground_points)}")
    return ground_points, non_ground_points

def load_components(path, cls):
    df = pq.read_table(path).to_pandas()
    return [cls.from_dict(r.to_dict()) for _, r in df.iterrows()]

def crop_image_with_box(img, box2d, margin=5, min_ratio=0.8):
    cx, cy = box2d.box.center.x, box2d.box.center.y
    w, h = box2d.box.size.x, box2d.box.size.y

    x_min, y_min = cx - w / 2, cy - h / 2
    x_max, y_max = cx + w / 2, cy + h / 2

    x_min_clip, y_min_clip = max(int(x_min), 0), max(int(y_min), 0)
    x_max_clip, y_max_clip = min(int(x_max), img.width), min(int(y_max), img.height)

    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    if ratio_inside < min_ratio:
        return None

    x_min_final = max(x_min_clip - margin, 0)
    y_min_final = max(y_min_clip - margin, 0)
    x_max_final = min(x_max_clip + margin, img.width)
    y_max_final = min(y_max_clip + margin, img.height)

    if x_max_final <= x_min_final or y_max_final <= y_min_final:
        return None

    return img.crop((x_min_final, y_min_final, x_max_final, y_max_final))

def crop_lidar_with_obb(pcd_points, box3d, min_points=5):
    if isinstance(pcd_points, o3d.geometry.PointCloud):
        pcd_np = np.asarray(pcd_points.points)
    else:
        pcd_np = np.asarray(pcd_points)

    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = np.array([box3d.box.center.x, box3d.box.center.y, box3d.box.center.z])
    obb.extent = np.array([box3d.box.size.x, box3d.box.size.y, box3d.box.size.z])
    obb.R = R.from_euler('z', box3d.box.heading).as_matrix()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)

    indices = obb.get_point_indices_within_bounding_box(pcd.points)
    cropped_points = np.asarray(pcd.points)[indices]

    if len(cropped_points) < min_points:
        return None
    
    return cropped_points
    
### Main Extraction Function ###
def extract_and_save_waymo_triplets(
    data_path,
    split="val",
    save_path="dataset/waymo_triplets/",
    min_points=15,
    margin=5,
    min_ratio=0.8,
    image_format="png",
    lidar_format="npy",
    sample_interval=2.0,
    segment_filter=None
):

    split_map = {'val': 'validation', 'train': 'training'} # Training split not publicly available
    original_split = split_map.get(split, 'validation')
    split_dir = os.path.join(data_path, original_split)
    save_jsonl = os.path.join(save_path, f"waymo_triplet_{split}.jsonl")
    image_dir = os.path.join(save_path, "waymo_image", split)
    lidar_dir = os.path.join(save_path, "waymo_lidar", split)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)

    lidar_paths = glob.glob(os.path.join(split_dir, "lidar", "*.parquet"))
    all_segments = sorted([os.path.basename(p).replace(".parquet", "") for p in lidar_paths])
    if segment_filter is not None:
        try:
            processed_filter = [int(x) if x.isdigit() else x for x in segment_filter]
        except ValueError:
            processed_filter = segment_filter            
        if all(isinstance(x, int) for x in processed_filter):
            # interpret as indices
            segment_ids = [all_segments[i] for i in processed_filter if 0 <= i < len(all_segments)]
            print(f"[INFO] Using segment indices {processed_filter}")
        elif all(isinstance(x, str) for x in processed_filter):
            # interpret as segment_id strings
            segment_ids = [s for s in all_segments if s in processed_filter]
            print(f"[INFO] Using segment IDs: {processed_filter}")
        else:
            raise ValueError("segment_filter must be a list of indices (int) or IDs (str)")
    else:
        segment_ids = all_segments
        print(f"[INFO] Found {len(segment_ids)} total segments in {split_dir}")

    total_saved = 0

    mode = "a" if os.path.exists(save_jsonl) else "w"
    with open(save_jsonl, mode, buffering=1) as f_out:
        lidar_freq = 10  # Waymo LiDAR：10Hz
        sample_step = int(sample_interval * lidar_freq)
        for seg_id in tqdm(segment_ids, desc="Processing segments"):
            written_segment = set()
            # === Load Waymo components ===
            cams = load_components(f"{split_dir}/camera_image/{seg_id}.parquet", CameraImageComponent)
            lidar_boxes = load_components(f"{split_dir}/lidar_box/{seg_id}.parquet", LiDARBoxComponent)
            proj_boxes = load_components(f"{split_dir}/projected_lidar_box/{seg_id}.parquet", ProjectedLiDARBoxComponent)
            lidars = load_components(f"{split_dir}/lidar/{seg_id}.parquet", LiDARComponent)
            calibs = load_components(f"{split_dir}/lidar_calibration/{seg_id}.parquet", LiDARCalibrationComponent)

            for frame_idx, lidar in enumerate(lidars):
                if frame_idx % sample_step != 0:
                    continue
                lidar_ts = lidar.key.frame_timestamp_micros
                calib = next((c for c in calibs if c.key.laser_name == lidar.key.laser_name), None)
                if calib is None:
                    continue

                # Range image to 3D point cloud
                pc1 = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, calib, frame_pose=None).numpy()[:, :3]
                pc2 = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return2, calib, frame_pose=None).numpy()[:, :3]
                points_vehicle = np.concatenate([pc1, pc2], axis=0)
                _, points_vehicle = segment_ground_o3d(points_vehicle)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_vehicle)

                # Match LiDAR boxes to Projected boxes
                lidar_boxes_frame = [b for b in lidar_boxes if b.key.frame_timestamp_micros == lidar_ts]
                proj_boxes_frame = [p for p in proj_boxes if p.key.frame_timestamp_micros == lidar_ts]

                # Deduplicate projected boxes
                unique_proj = {}
                for pb in proj_boxes_frame:
                    key = (pb.key.segment_context_name, pb.key.frame_timestamp_micros, pb.key.laser_object_id, pb.key.camera_name)
                    if key not in unique_proj:
                        unique_proj[key] = pb
                proj_boxes_frame = list(unique_proj.values())

                lidar_dict = {
                    (b.key.segment_context_name, b.key.frame_timestamp_micros, b.key.laser_object_id): b
                    for b in lidar_boxes_frame
                }

                for pb in proj_boxes_frame:
                    key = (pb.key.segment_context_name, pb.key.frame_timestamp_micros, pb.key.laser_object_id)
                    if key not in lidar_dict:
                        continue

                    box3d = lidar_dict[key]
                    box2d = pb
                    cam_name = box2d.key.camera_name

                    # Load image
                    cam_frame = [
                        c for c in cams
                        if c.key.camera_name == cam_name and c.key.frame_timestamp_micros == box2d.key.frame_timestamp_micros
                    ]
                    if not cam_frame:
                        continue
                    img = Image.open(BytesIO(cam_frame[0].image)).convert("RGB")

                    # Crop image and lidar
                    cropped_img = crop_image_with_box(img, box2d, margin=margin, min_ratio=min_ratio)
                    if cropped_img is None:
                        continue

                    lidar_points = crop_lidar_with_obb(np.asarray(pcd.points), box3d, min_points=min_points)
                    if lidar_points is None:
                        continue

                    # Save outputs
                    label_id = box3d.type
                    label = WAYMO_LABEL_MAP.get(label_id, "Unknown")
                    if label == "Unknown":
                        continue

                    obj_id = box3d.key.laser_object_id
                    base_name = f"{seg_id}_{lidar_ts}_{cam_name}_{obj_id}_{label}"
                    img_filename = f"{base_name}.{image_format}"
                    lidar_filename = f"{base_name}.{lidar_format}"
                    img_path = os.path.join(image_dir, img_filename)
                    lidar_path = os.path.join(lidar_dir, lidar_filename)
                    img_rel_path = os.path.join("waymo_image", split, img_filename)
                    lidar_rel_path = os.path.join("waymo_lidar", split, lidar_filename)

                    # skip duplicates
                    write_key = (seg_id, lidar_ts, cam_name, obj_id)
                    if write_key in written_segment:
                        continue
                    written_segment.add(write_key)

                    cropped_img.save(img_path)
                    np.save(lidar_path, lidar_points)

                    bbox = [
                        box3d.box.center.x, box3d.box.center.y, box3d.box.center.z,
                        box3d.box.size.x, box3d.box.size.y, box3d.box.size.z, box3d.box.heading
                    ]

                    entry = {
                        "segment_id": seg_id,
                        "timestamp": lidar_ts,
                        "label_id": label_id,
                        "label": label,
                        "image_path": img_rel_path,
                        "lidar_path": lidar_rel_path,
                        "bbox": bbox,
                        "camera_name": cam_name,
                        "laser_object_id": obj_id
                    }

                    f_out.write(json.dumps(entry) + "\n")
                    total_saved += 1

    print(f"\n [DONE] Extracted and saved {total_saved} triplets -> {save_jsonl}")
    print(f"Images saved to: {image_dir}")
    print(f"LiDARs saved to: {lidar_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract text-lidar-image triplets from Waymo Open Dataset (v2) and save as JSONL."
    )

    paths = parser.add_argument_group('Paths')
    paths.add_argument("--data_path", type=str, required=True,
                       help="Path to the Waymo Open Dataset root folder.")
    paths.add_argument("--save_path", type=str, default="dataset/waymo_triplets/",
                       help="Folder to save the output dataset file.")
    paths.add_argument("--split", type=str, choices=['val'], 
                       default='val', help="Dataset split to process.")
    quality = parser.add_argument_group('Quality Filters')
    quality.add_argument("--min_points", type=int, default=15,
                         help="Min LiDAR points inside a 3D box to be valid.")
    quality.add_argument("--min_ratio", type=float, default=0.8,
                         help="Min visibility ratio of 2D box inside image.")
    quality.add_argument("--margin", type=int, default=5,
                         help="Pixel margin to add around cropped image boxes.")
    sampling = parser.add_argument_group('Sampling')
    sampling.add_argument("--sample_interval", type=float, default=4.0, 
                          help="Time interval (seconds) to sample frames (1 segment ≈ 20s).")
    sampling.add_argument("--segment_filter", type=str, nargs='*', default=None,
                          help="List of segment IDs/indices to process. Example: --segment_filter {0..49}")
    formats = parser.add_argument_group('Output Formats')
    formats.add_argument("--image_format", type=str, choices=['png', 'jpg'], default='png',
                         help="Format for saved image crops.")
    formats.add_argument("--lidar_format", type=str, choices=['npy'], default='npy',
                         help="Format for saved LiDAR point clouds.")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"[INFO] Starting Waymo Triplet Extraction")
    print(f"[INFO] Destination: {args.save_path}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Sampling Interval: {args.sample_interval}s")
    print(f"{'='*60}")

    try:
        extract_and_save_waymo_triplets(
            data_path=args.data_path,
            split=args.split,
            save_path=args.save_path,
            min_points=args.min_points,
            margin=args.margin,
            min_ratio=args.min_ratio,
            sample_interval=args.sample_interval,
            segment_filter=args.segment_filter
        )
        print("\n[SUCCESS] Extraction completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")

if __name__ == "__main__":
    main()
