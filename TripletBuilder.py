import argparse
from pathlib import Path
from data.kitti_triplet import KITTI_TripletDataset
from utils.data_utils import save_triplet_dataset_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text-lidar-image triplets from KITTI and save as JSONL.")
    parser.add_argument("--data_path", type=str, default="/home/ximeng/Dataset/kitti/",
                        help="Path to the KITTI dataset root folder.")
    parser.add_argument("--save_path", type=str, default="data/",
                        help="Folder to save the output dataset file.")
    parser.add_argument("--image_format", type=str, choices=['png', 'jpg'], default='png',
                        help="Image format to save (png or jpg).")
    parser.add_argument("--image_dir", type=str, default="kitti_images",
                        help="Subfolder within save_path to store images.")
    parser.add_argument("--split", type=str, choices=['training', 'testing'], default='training',
                        help="KITTI dataset split (training or testing).")
    parser.add_argument("--min_points", type=int, default=15,
                        help="Minimum number of LiDAR points required inside a 3D bounding box.")

    args = parser.parse_args()

    print(f"[INFO] Loading KITTI Triplet Dataset from {args.data_path}/{args.split} ...")
    dataset = KITTI_TripletDataset(
        root_dir=args.data_path,
        split=args.split,
        min_points=args.min_points
    )
    print(f"[INFO] Total frames in {args.split}: {len(dataset)}")

    save_folder = Path(args.save_path)
    save_folder.mkdir(parents=True, exist_ok=True)

    split_name = "train" if args.split == "training" else "test"
    save_path = save_folder / f"kitti_triplet_{split_name}.jsonl"

    image_dir = save_folder / args.image_dir / split_name
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving KITTI {split_name} triplets to {save_path}")
    print(f"[INFO] Image outputs will be saved under {image_dir}")

    save_triplet_dataset_jsonl(
        dataset,
        save_jsonl_path=str(save_path),
        image_dir=str(image_dir),
        split=split_name,
        image_format=args.image_format
    )

    print(f"[SUCCESS] KITTI {split_name} triplet dataset extraction completed")
