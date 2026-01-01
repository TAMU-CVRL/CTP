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

```py
python3 ./eval_align.py --config configs/ctp_default.yaml --eval_path dataset/nuscenes_triplets/nuscenes_triplet_train.jsonl --after_ckpt checkpoints/ckpt_epoch9.pt --label truck
```

## Environment

Base

```
conda create -n ctp python=3.9

conda activate ctp

pip install pyyaml typeguard

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install git+https://github.com/openai/CLIP.git

pip install tensorboard wandb

pip install transformers

pip install matplotlib

pip install nuscenes-devkit
```

PTv3
```
pip install timm

pip install torch-scatter

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

pip install spconv-cu126
```

Flash Attention
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

```
python3 train.py --config configs/ctp_pn2.yaml

python3 train.py --config configs/ctp_ptv3.yaml
```