# Contrastive Tensor Pre-training (CTP)

## NuScenes
```
python3 ./TripletBuilder.py --dataset nuscenes --data_path /home/ximeng/Dataset/nuscenes_full_v1_0/ --split train
```

## KITTI
```py
python3 ./TripletBuilder.py --dataset kitti --data_path /home/ximeng/Dataset/kitti/
```

## Waymo
```py
python3 ./TripletBuilder_waymo.py --data_path /home/ximeng/Dataset/waymo_open_dataset_v_2_0_1/ --segment_filter {0..49}
```

##
```py
python3 ./CaptionGen.py --jsonl_path dataset/nuscenes_triplets/nuscenes_triplet_train.jsonl
```
```py
python3 ./CaptionGen.py --jsonl_path dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl
```
```py
python3 ./CaptionGen.py --jsonl_path dataset/kitti_triplets/kitti_triplet_train.jsonl
```
```py
python3 ./CaptionGen.py --jsonl_path dataset/waymo_triplets/waymo_triplet_val.jsonl
```

## Train
```py
python3 ./train.py
```

## Eval
```py
python3 ./CTPEvaluator.py --config configs/ctp_default.yaml --eval_path dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl
```

```py
python3 ./CTPEvaluator.py --config configs/ctp_default.yaml --eval_path dataset/kitti_triplets/kitti_triplet_train.jsonl
```

```py
python3 ./CTPEvaluator.py --config configs/ctp_default.yaml --eval_path dataset/waymo_triplets/waymo_triplet_val.jsonl
```