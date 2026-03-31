# Toward Unified Multimodal Representation Learning for Autonomous Driving
[![arXiv](https://img.shields.io/badge/arXiv-2603.07874-b31b1b.svg)](https://arxiv.org/abs/2603.07874)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Ximeng0831/CTP)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Ximeng0831/CTP-Dataset)
[![GitHub](https://img.shields.io/badge/GitHub-CTP-lightgrey?logo=github)](https://github.com/TAMU-CVRL/CTP)

Overview of Contrastive Tensor Pre-training (CTP). We propose this framework that simultaneously aligns
multiple modalities in a similarity tensor.

![pipeline](./figures/pipeline.jpg)
<!-- <img src="./figures/pipeline.jpg" alt="overview" width="1000" align="center" /> -->
For the “car” class, we project features from three modalities using 200 samples onto a 2D plane.
![result](./figures/align_comparison.jpg)
To run a quick inference demo, clone the repository, set up the environment, and execute the notebook located at [`demo/inference.ipynb`](https://github.com/TAMU-CVRL/CTP/blob/main/demo/inference.ipynb).

## Requirements

### Option 1: Automatic Setup

The provided script can be used to set up the environment:

```bash
bash ./scripts/setup_env.sh
```

### Option 2: Manual Setup

We can create a [conda](https://docs.conda.io/en/latest/) environment named `ctp`:
``` bash
conda create -n ctp python=3.10
```
Then activate the environment and install required libraries:
``` bash
conda activate ctp
```
Install [PyTorch](https://pytorch.org/get-started/locally/) based on your GPU:
``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Install other libraries:
``` bash
pip install transformers==5.1.0 nuscenes-devkit==1.2.0 pandas==2.3.3 open3d==0.19.0 wandb==0.25.1 tensorboard==2.20.0 git+https://github.com/openai/CLIP.git matplotlib==3.9.4 huggingface_hub==1.8.0 umap-learn==0.5.11 accelerate==1.13.0 beautifulsoup4==4.14.3 typeguard==4.4.4 pyyaml==6.0.3 tqdm==4.67.1 idna==3.11 ipykernel==7.2.0 ipywidgets==8.1.8 pickleshare==0.7.5 jmespath==1.1.0 pyrootutils==1.0.4
```

## Triplet Data Preparation
Triplet data preparation can be divided into two steps. First, we extract annotations, cropped images, and the corresponding point clouds. Then, the images and annotations are fed into a VLM ([Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)) to generate pseudo captions.

### Dataset Structure
```text
dataset/
├── nuscenes_triplets/
│   ├── nuscenes_image/
│   ├── nuscenes_lidar/
│   ├── nuscenes_triplet_train.jsonl
│   └── nuscenes_triplet_val.jsonl
├── kitti_triplets/
│   ├── kitti_image/
│   ├── kitti_lidar/
│   └── kitti_triplet_train.jsonl
└── waymo_triplets/
    ├── waymo_image/
    ├── waymo_lidar/
    └── waymo_triplet_val.jsonl
```
<!-- ### Metadata Format
Each line in the `.jsonl` file represents a single triplet sample. For example:
```json
{
  "label": "trafficcone",
  "image_path": "nuscenes_image/val/val_0_0_trafficcone.png",
  "lidar_path": "nuscenes_lidar/val/val_0_0_trafficcone.npy",
  "bbox": [0.966, -5.245, 0.659, 0.291, 0.302, 1.265, 1.551],
  "caption": "The traffic cone is orange with a white reflective band near the top, has a conical geometry tapering to a point, and features a black and yellow reflective strip near its base."
}
``` -->
The dataset is available on Hugging Face. For more details, please refer to the [dataset page](https://huggingface.co/datasets/Ximeng0831/CTP-Dataset).

### Training Triplet Dataset
``` bash
python3 ./TripletBuilder.py --dataset nuscenes --data_path /PATH/TO/NUSCENES/DATASET --split train
```

```bash
python3 ./CaptionGen.py --jsonl_path dataset/nuscenes_triplets/nuscenes_triplet_train.jsonl
```
### Test Triplet Dataset
- NuScenes
``` bash
python3 ./TripletBuilder.py --dataset nuscenes --data_path /PATH/TO/NUSCENES/DATASET --split val
```
``` bash
python3 ./CaptionGen.py --jsonl_path dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl
```
- KITTI
``` bash
python3 ./TripletBuilder.py --dataset kitti --data_path /PATH/TO/KITTI/DATASET
```
``` bash
python3 ./CaptionGen.py --jsonl_path dataset/kitti_triplets/kitti_triplet_train.jsonl
```
- Waymo

To generate the Waymo triplet dataset, first create a separate environment named `waymo`:
``` bash
conda create -n waymo python=3.10
conda activate waymo
```
Install the required dependencies:
``` bash
pip install numpy pandas pyarrow pillow tqdm scipy open3d waymo-open-dataset-tf-2-12-0
```
Then generate the triplet data and pseudo captions:
```  bash
python3 ./TripletBuilder_waymo.py --data_path /PATH/TO/WMOD/DATASET --segment_filter {0..49}
```
``` bash
python3 ./CaptionGen.py --jsonl_path dataset/waymo_triplets/waymo_triplet_val.jsonl
```
After finishing the data generation, we can switch back to the `ctp` environment:
``` bash
conda activate ctp
```
## Training Models
To train a model, simply provide a configuration file. The configuration files can be modified in the `./configs` folder.
``` bash
python3 ./train.py --config configs/default.yaml
```
Configuration Options:
- **masked** (`True` / `False`): Whether to use the masking strategy.
- **pc_only** (`True` / `False`): Whether to train only the point cloud encoder or all encoders.
- **use_tb** (`True` / `False`): Whether to enable TensorBoard logging.
- **use_wandb** (`True` / `False`): Whether to enable Weights & Biases logging. Run `wandb login` first to authenticate.

Checkpoints are available on [Hugging Face](https://huggingface.co/Ximeng0831/CTP).

## Evaluation
### Zero-shot Classification Accuracy
To evaluate a trained model, first set **`checkpoint_path`** in the configuration file. Then choose an evaluation dataset from the following options:

- `dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl`
- `dataset/kitti_triplets/kitti_triplet_train.jsonl`
- `dataset/waymo_triplets/waymo_triplet_val.jsonl`

For example:

```bash
python3 ./eval_acc.py --config configs/default.yaml --eval_path dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl --tau 0.5
```
The parameter **`tau`** controls modality usage during evaluation:

- **`tau = 0`**: Only the point cloud modality is used.
- **`tau = 1`**: Only the image modality is used.
- **`tau = 0.5`**: Both modalities are jointly evaluated.
### Alignment
To evaluate the alignment effect, high-dimensional features are projected onto a 2D plane to compare representations before and after alignment.

Run the example command:
``` bash
python3 ./eval_align.py --config configs/default.yaml --eval_path dataset/nuscenes_triplets/nuscenes_triplet_train.jsonl --after_ckpt PATH/TO/CHECKPOINT --label car
```
Arugments:
- **`--eval_path`**  
  Supported evaluation datasets:

  - `dataset/nuscenes_triplets/nuscenes_triplet_val.jsonl`
  - `dataset/kitti_triplets/kitti_triplet_train.jsonl`
  - `dataset/waymo_triplets/waymo_triplet_val.jsonl`

- **`--after_ckpt`**: Path to the checkpoint file you want to evaluate.

- **`--label`**: Object category used to visualize alignment effects. Supported labels include:
  - `"car"`
  - `"truck"`
  - `"pedestrian"`

## BibTeX
If you find this work useful for your research, please cite our work:
```
@misc{tao2026ctp,
      title={Toward Unified Multimodal Representation Learning for Autonomous Driving}, 
      author={Ximeng Tao and Dimitar Filev and Gaurav Pandey},
      year={2026},
      eprint={2603.07874},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.07874}, 
}
```
