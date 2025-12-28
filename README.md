# Contrastive Tensor Pre-training (CTP)

## NuScenes
```
python3 TripletBuilder.py --dataset nuscenes --data_path /home/ximeng/Dataset/nuscenes_full_v1_0/ --split train
```

## KITTI
```
python3 TripletBuilder.py --dataset kitti --data_path /home/ximeng/Dataset/kitti/
```

## Waymo
```
python3 ./TripletBuilder_waymo.py --data_path /home/ximeng/Dataset/waymo_open_dataset_v_2_0_1/ --segment_filter {0..49}
```

##
```
python3 CaptionGen.py --jsonl_path dataset/nuscenes_triplets/nuscenes_triplet_train.jsonl
```
```
python3 CaptionGen.py --jsonl_path dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl
```
```
python3 CaptionGen.py --jsonl_path dataset/kitti_triplets/kitti_triplet_train.jsonl
```
```
python3 CaptionGen.py --jsonl_path dataset/waymo_triplets/waymo_triplet_val.jsonl
```

##
```
python3 ./train.py
```
