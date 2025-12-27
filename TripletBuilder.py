import argparse
from pathlib import Path
from utils.data_utils import save_triplet_dataset_jsonl
# nuScenes
from nuscenes.nuscenes import NuScenes
from data.nuscenes_data import NuscenesData
from data.nuscenes_triplet import Nuscenes_TripletDataset
# KITTI
from data.kitti_triplet import KITTI_TripletDataset

def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Extractor for NuScenes and KITTI Triplets")

    # --- Common Arguments ---
    parser.add_argument("--dataset", type=str, choices=['nuscenes', 'kitti'], required=True,
                        help="Target dataset to process: 'nuscenes' or 'kitti'")
    parser.add_argument("--data_path", type=str, default="/path/to/dataset/",
                        help="Root directory of the raw dataset.")
    parser.add_argument("--save_path", type=str, default="dataset/",
                        help="Base directory to save output JSONL and images.")
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'], default='train',
                        help="Dataset split to process.")
    parser.add_argument("--image_format", type=str, choices=['png', 'jpg'], default='png',
                        help="Format for exported images.")
    parser.add_argument("--lidar_format", type=str, choices=['npy', 'txt'], default='npy',
                        help="Format for exported LiDAR data.")

    # --- NuScenes Specific Arguments ---
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        help="[NuScenes] Dataset version (e.g., v1.0-mini, v1.0-trainval)")
    parser.add_argument("--nusc_img_dir", type=str, default="nuscenes_images",
                        help="[NuScenes] Subfolder name for saved images.")
    parser.add_argument("--nusc_lidar_dir", type=str, default="nuscenes_lidar",
                        help="[NuScenes] Subfolder name for NuScenes LiDAR data.")
    
    # --- KITTI Specific Arguments ---
    parser.add_argument("--min_points", type=int, default=15,
                        help="[KITTI] Min LiDAR points required inside a 3D bounding box.")
    parser.add_argument("--kitti_img_dir", type=str, default="kitti_images",
                        help="[KITTI] Subfolder name for saved images.")
    parser.add_argument("--kitti_lidar_dir", type=str, default="kitti_lidar",
                        help="[KITTI] Subfolder name for KITTI LiDAR data.")

    args = parser.parse_args()

    # Create base save directory
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    split_map = {'train': 0, 'val': 1, 'test': 2}
    split_idx = split_map[args.split]
    # ---------------------------------------------------------
    # NuScenes Dataset
    # ---------------------------------------------------------
    if args.dataset == 'nuscenes':
        print(f"[INFO] Initializing NuScenes ({args.version}) from {args.data_path}...")
        nusc = NuScenes(version=args.version, dataroot=args.data_path)
        
        # Initialize NuScenes data loaders
        raw_dataset = NuscenesData(nusc, split_idx, pre_frames=0, future_frames=0)
        wrapper_dataset = Nuscenes_TripletDataset(raw_dataset)

        # Setup output paths
        save_folder = save_path / "nuscenes_triplets"
        save_folder.mkdir(parents=True, exist_ok=True)

        rel_img_path = Path(args.nusc_img_dir) / args.split
        rel_lidar_path = Path(args.nusc_lidar_dir) / args.split
        jsonl_save_path = save_folder / f"nuscenes_triplet_{args.split}.jsonl"

        print(f"[INFO] Loaded {len(wrapper_dataset)} samples. Starting extraction...")
        save_triplet_dataset_jsonl(
            wrapper_dataset,
            save_jsonl_path=str(jsonl_save_path),
            rel_image_dir=str(rel_img_path),
            rel_lidar_dir=str(rel_lidar_path),
            split=args.split,
            image_format=args.image_format,
            lidar_format=args.lidar_format
        )

    # ---------------------------------------------------------
    # KITTI Dataset
    # ---------------------------------------------------------
    elif args.dataset == 'kitti':
        # Map 'train/val/test' to KITTI folder structure ('training' or 'testing')
        kitti_split_dir = 'testing' if split_idx == 2 else 'training'
        
        print(f"[INFO] Loading KITTI Dataset from {args.data_path}/{kitti_split_dir}...")
        dataset = KITTI_TripletDataset(
            root_dir=args.data_path,
            split=kitti_split_dir,
            min_points=args.min_points
        )

        # Setup output paths
        save_folder = save_path / "kitti_triplets"
        save_folder.mkdir(parents=True, exist_ok=True)

        rel_img_path = Path(args.kitti_img_dir) / args.split
        rel_lidar_path = Path(args.kitti_lidar_dir) / args.split
        jsonl_save_path = save_folder / f"kitti_triplet_{args.split}.jsonl"

        print(f"[INFO] Total frames in {args.split}: {len(dataset)}. Starting extraction...")
        save_triplet_dataset_jsonl(
            dataset,
            save_jsonl_path=str(jsonl_save_path),
            rel_image_dir=str(rel_img_path),
            rel_lidar_dir=str(rel_lidar_path),
            split=args.split,
            image_format=args.image_format,
            lidar_format=args.lidar_format
        )

    print(f"[SUCCESS] Extraction for {args.dataset.upper()} ({args.split}) finished.")

if __name__ == "__main__":
    main()
