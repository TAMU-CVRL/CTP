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