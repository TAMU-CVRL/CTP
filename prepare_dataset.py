
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from data.nuscenes_data import NuscenesData
from data.nuscenes_wrapper import SparseCLIP_Dataset, save_triplet_dataset_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text-lidar pairs from NuScenes and save as dataset.")
    parser.add_argument("--data_path", type=str, default="/home/ximeng/Dataset/nuscenes_full_v1_0/",
                        help="Path to the NuScenes dataset root folder.")
    parser.add_argument("--save_path", type=str, default="data/",
                        help="Folder to save the output dataset file.")
    parser.add_argument("--image_format", type=str, choices=['png', 'jpg'], default='png',
                        help="Image format to save (png or jpg).")
    parser.add_argument("--image_dir", type=str, default="nuscenes_images",
                        help="Subfolder within save_path to store images.")
    parser.add_argument("--is_train", type=int, choices=[0, 1], default=0,
                        help="Dataset split: 0 = train, 1 = val")
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        help="NuScenes version, e.g., v1.0-mini or v1.0-trainval")

    args = parser.parse_args()

    # load nuscenes dataset
    data_path = Path(args.data_path)
    nusc = NuScenes(version=args.version, dataroot=data_path)

    is_train = args.is_train # 0: train, 1: val
    pre_frame = 0
    future_frame = 0
    dataset = NuscenesData(nusc, is_train, pre_frame, future_frame)
    SparseCLIP_dataset = SparseCLIP_Dataset(dataset)

    split = 'train' if is_train == 0 else 'val'

    print(f"Loaded NuScenes {split} dataset with {len(SparseCLIP_dataset)} samples.")

    # save dataset
    save_folder = Path(args.save_path)
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / f"nuscenes_triplet_{split}"
    image_dir = save_folder / args.image_dir / split  # e.g., data/images/train
    image_dir.mkdir(parents=True, exist_ok=True)

    save_triplet_dataset_jsonl(
        SparseCLIP_dataset,
        save_jsonl_path=str(save_path) + ".jsonl",
        image_dir=str(image_dir),
        split=split,
        image_format=args.image_format
    )
