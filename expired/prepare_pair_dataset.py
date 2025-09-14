
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from data.nuscenes_data import NuscenesData
from data.nuscenes_wrapper import SparseCLIP_Dataset
from functions import TextLiDARPairDataset, save_text_lidar_pair_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text-lidar pairs from NuScenes and save as dataset.")
    parser.add_argument("--data_path", type=str, default="/home/ximeng/Dataset/nuscenes_full_v1_0/",
                        help="Path to the NuScenes dataset root folder.")
    parser.add_argument("--save_path", type=str, default="data/",
                        help="Folder to save the output dataset file.")
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

    if is_train == 0:
        split = 'train'
    elif is_train == 1:
        split = 'val'
    else:
        split = 'test'

    print(f"Loaded NuScenes {split} dataset with {len(SparseCLIP_dataset)} samples.")

    # save dataset
    save_folder = Path(args.save_path)
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / f"nuscenes_tp_{split}.pt"
    dataset = save_text_lidar_pair_dataset(SparseCLIP_dataset, save_path=save_path)
