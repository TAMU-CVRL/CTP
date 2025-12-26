import os
import argparse
import math
import clip
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from models.ctp import sclip_triplet
from models.pointnet2 import pointnet2_encoder
from data.nuscenes_triplet import Triplet_Object
from utils.img_utils import image_transform
from utils.pc_utils import load_sparse_method
from utils.model_utils import get_clip_encoders, prepare_training, load_config, build_scheduler, gather_features

import wandb

### Training ###
def train(model, train_dataloader, optimizer, scheduler, cfg):
    # Load configurations
    step = cfg.get("start_step", 0)
    start_epoch = cfg.get("start_epoch", 0)
    epochs = cfg.get("epochs", 10)
    device = cfg.get("device", "cuda")
    accumulation_steps = cfg.get("accumulation_steps", 1)
    tensorboard_logdir = cfg.get("tensorboard_logdir", None)
    rank = cfg.get("rank", 0)
    world_size = cfg.get("world_size", 1)
    detailed_caption = cfg.get("detailed_caption", False)

    # Checkpoint and TensorBoard
    if rank == 0:
        wandb.config.update({
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"],
            "warmup_ratio": cfg["warmup_ratio"],
            "accumulation_steps": accumulation_steps,
            "optimizer": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "epochs": cfg["epochs"],
            "scheduler": type(scheduler).__name__,
        })

        os.makedirs(f"checkpoints/{cfg['name']}", exist_ok=True)

        if tensorboard_logdir is not None:
            train_writer = SummaryWriter(log_dir=tensorboard_logdir)
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = f"runs/{current_time}_{cfg['name']}"
            train_writer = SummaryWriter(log_dir=log_dir)
    else:
        train_writer = None  # Only rank 0 writes to TensorBoard
    
    # Training
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        updates_in_epoch = 0

        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(train_dataloader, total=len(train_dataloader))
        else:
            pbar = train_dataloader

        for i, sample in enumerate(pbar):
            # Get data
            if detailed_caption:
                caption = sample["caption"]
            else:
                caption = sample["label"]
            imgs = sample["image"].to(device) # [B, 3, H, W]
            points = sample["lidar"].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
            text_ids = clip.tokenize(caption, truncate=True).to(device) # str -> tensor

            # Calculate loss
            text_features, image_features, lidar_features = model(text_ids, imgs, points)
            text_features, image_features, lidar_features = gather_features(text_features, image_features, lidar_features, gather_with_grad=True, world_size=world_size)
            core = model.module if hasattr(model, "module") else model
            loss, _ = core.get_loss(text_features, image_features, lidar_features)
            
            # Backprop
            original_loss = loss.item()
            loss = loss / accumulation_steps # scale loss
            loss.backward() # calculate gradients

            # Accumulate gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                # Update model
                optimizer.step() # update model
                optimizer.zero_grad() # reset gradients
                scheduler.step() # update learning rate
                step += 1
                updates_in_epoch += 1 

                # Log to TensorBoard
                running_loss += original_loss # accumulate loss
                if rank == 0:
                    train_writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], step)
                    wandb.log({"Loss/train_step": original_loss, "Learning Rate": optimizer.param_groups[0]["lr"]}, step=step)
                    if original_loss != 0: # final tiny batch will cause loss = 0
                        train_writer.add_scalar("Loss/train_step", original_loss, step)
            # Log to console
            if rank == 0 and step % 2 == 0:
                pbar.set_description(f"Epoch {epoch} | Step {step} | Loss: {original_loss:.4f}")
        
        # Log epoch loss
        if dist.is_initialized(): 
            dist.barrier()
        if rank == 0:
            train_avg_loss = running_loss / max(updates_in_epoch, 1)
            train_writer.add_scalar("Loss/train_epoch", train_avg_loss, epoch)
            wandb.log({"Loss/train_epoch": train_avg_loss, "Epoch": epoch}, step=step)
            print(f"Epoch {epoch} | Train Loss: {train_avg_loss:.4f}")
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"checkpoints/{cfg['name']}/ckpt_epoch{epoch}.pt")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
###############################################################################
# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SparseCLIP model")
    parser.add_argument("--config_path", type=str, default="path/to/config.yaml",
                        help="Path to the config file.")
    args = parser.parse_args()
    cfg = load_config(args.config_path)

    ### Initialize distributed ###
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"DDP initialized: rank {rank}/{world_size}, local_rank {local_rank}")
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1
        print(f"Single-GPU training on {device}")

    ### Load dataset ###
    batch_size = cfg["Train"]["batch_size"]
    sparse_method = cfg["Dataset"]["sparse_method"]
    sparse_to_dense = load_sparse_method(sparse_method)

    tmpdir = os.getenv("TMPDIR", None)

    if tmpdir and os.path.exists(tmpdir):
        tar_path = os.path.join(tmpdir, "nuscenes_images.tar")
        if os.path.exists(tar_path):
            print(f"[INFO] Using local NVMe dataset at {tar_path}")
        else:
            print(f"[WARN] $TMPDIR found but tar not copied. Using YAML path instead.")
            tar_path = cfg["Dataset"]["image_tar_path"]
    else:
        print(f"[INFO] $TMPDIR not set. Using YAML path: {cfg['Dataset']['image_tar_path']}")
        tar_path = cfg["Dataset"]["image_tar_path"]

    train_dataset = Triplet_Object(
        jsonl_file=cfg["Dataset"]["train_data_path"],
        image_transform=image_transform,
        sparse_to_dense_fn=sparse_to_dense,
        image_tar_path=tar_path
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
        shuffle_train = False  # DDP requires sampler, so disable shuffle
    else:
        train_sampler = None
        eval_sampler = None
        shuffle_train = True  # Enable shuffle for single-GPU
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=cfg["Train"].get("num_workers", 0),
        pin_memory=True
    )

    ### Load models ###
    # Clip text and image encoder
    pc_only = cfg["Model"]["pc_only"]

    # Initialize W&B
    if rank == 0:
        wandb.init(
            project="ctp",
            name=cfg["Name"],
            mode="offline",
            monitor_gym=False
        )

    clip_model_name = cfg["Model"]["clip_model"]
    if pc_only:
        # Clip trained encoder
        clip_model, clip_preprocess = clip.load(clip_model_name, jit=False)
        clip_model.eval()
        clip_model.to(device)
        for param in clip_model.parameters():
            param.requires_grad = False
        text_encoder = clip_model.encode_text
        img_encoder = clip_model.encode_image
        print(f"Loading trained encoder from {clip_model_name}")
    else:
        # Clip untrained encoder
        text_encoder, img_encoder = get_clip_encoders(clip_model_name)
        text_encoder.to(device)
        img_encoder.to(device)
        print(f"Loading untrained encoder from {clip_model_name}")
    # PointNet++ encoder
    lidar_encoder = pointnet2_encoder.PointNet2Encoder() # [B, C, N] -> [B, C']
    lidar_encoder.to(device)
    # model
    model = sclip_triplet(
        text_encoder=text_encoder,
        image_encoder=img_encoder,
        lidar_encoder=lidar_encoder,
        loss_fn=cfg["Model"]["loss_fn"],
        alpha=cfg["Model"]["alpha"]
        ).to(device)

    if pc_only:
        # LiDAR + projector
        optimizer = optim.AdamW(
            list(model.lidar_encoder.parameters()) + list(model.projector.parameters()),
            lr=cfg["Train"]["learning_rate"],
            weight_decay=cfg["Train"]["weight_decay"]
        )
    else:
        # Text + image + LiDAR + projector
        optimizer = optim.AdamW(
            list(model.text_encoder.parameters()) +
            list(model.image_encoder.parameters()) +
            list(model.lidar_encoder.parameters()) +
            list(model.projector.parameters()),
            lr=cfg["Train"]["learning_rate"],
            weight_decay=cfg["Train"]["weight_decay"]
        )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ### Training configuration ###
    epochs = cfg["Train"]["epochs"]
    accumulation_steps = cfg["Train"]["accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    total_training_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg["Train"]["warmup_ratio"] * total_training_steps)

    scheduler_cfg = cfg["Train"]["scheduler"]
    scheduler = build_scheduler(optimizer, scheduler_cfg, warmup_steps, total_training_steps)

    checkpoint_path = cfg["Train"].get("checkpoint_path", None)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
    else:
        start_epoch = 0
        step = 0
        print("No checkpoint provided, training from scratch.")

    ### Train the model ###
    train_config = {
        "name": cfg["Name"],
        "accumulation_steps": accumulation_steps,
        "start_epoch": start_epoch,
        "start_step": step,
        "epochs": epochs,
        "device": device,
        "tensorboard_logdir": cfg["Train"].get("tensorboard_logdir", None),
        "rank": rank,
        "world_size": world_size,
        "detailed_caption": cfg["Dataset"].get("detailed_caption", False),
        "batch_size": cfg["Train"]["batch_size"],
        "weight_decay": cfg["Train"]["weight_decay"],
        "warmup_ratio": cfg["Train"]["warmup_ratio"],
    }
    prepare_training()
    train(model, train_dataloader, optimizer, scheduler, train_config)
    