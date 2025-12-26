import os
import yaml
import argparse
import clip

from tqdm import tqdm
from datetime import datetime

from models.ctp import sclip_triplet
from models.pointnet2 import pointnet2_encoder
from models.clip.clip import tokenize
from data.nuscenes_triplet import Triplet_Object
from data.nuscenes_wrapper import TripletJsonlDataset
from utils.processing import load_sparse_method, image_transform
from utils.model_utils import get_clip_encoders, prepare_training, load_config, build_scheduler

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

### Training ###
def train(model, train_dataloader, eval_dataloader, optimizer, scheduler, cfg):
    step = cfg.get("start_step", 0)
    start_epoch = cfg.get("start_epoch", 0)
    epochs = cfg.get("epochs", 10)
    device = cfg.get("device", "cuda")
    rank = cfg.get("rank", 0)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    
    tensorboard_logdir = cfg.get("tensorboard_logdir", None)

    if rank == 0:
        os.makedirs(f"checkpoints/{cfg['name']}", exist_ok=True)

        if tensorboard_logdir is not None:
            train_writer = SummaryWriter(log_dir=tensorboard_logdir)
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = f"runs/{current_time}_{cfg['name']}"
            train_writer = SummaryWriter(log_dir=log_dir)
    else:
        train_writer = None  # Only rank 0 writes to TensorBoard
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(train_dataloader, total=len(train_dataloader))
        else:
            pbar = train_dataloader

        for i, sample in enumerate(pbar):
            # Get data
            labels = sample[0]   # string label
            imgs = sample[1].to(device) # [B, 3, H, W]
            points = sample[2].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
            text_ids = clip.tokenize(labels, truncate=True).to(device) # str -> tensor

            # Calculate loss           
            text_features, image_features, lidar_features = model(text_ids, imgs, points)
            loss, _ = model.get_loss(text_features, image_features, lidar_features)
            
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

                # Log to TensorBoard only on rank 0
                running_loss += original_loss # accumulate loss
                if rank == 0:
                    train_writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], step)
                    if original_loss != 0: # final tiny batch will cause loss = 0
                        train_writer.add_scalar("Loss/train_step", original_loss, step)
            # Log to console
            if rank == 0 and step % 2 == 0:
                pbar.write(f"[LOG] Epoch {epoch} Step {step} Loss {(original_loss):.4f}")
        
        # Log epoch loss
        train_avg_loss = running_loss / len(train_dataloader)
        if rank == 0:
            train_writer.add_scalar("Loss/train_epoch", train_avg_loss, epoch)

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_avg_loss = evaluate(model, eval_dataloader, device, rank=rank, world_size=cfg.get("world_size", 1))

        if rank == 0:
            train_writer.add_scalar("Loss/val_epoch", val_avg_loss, epoch)
        
            train_writer.add_scalars("Loss/epoch", {
                "train": train_avg_loss,
                "val": val_avg_loss
            }, epoch)
            print(f"Epoch {epoch} | Train Loss: {train_avg_loss:.4f} | Validation Loss: {val_avg_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"checkpoints/{cfg['name']}/ckpt_epoch{epoch}.pt")

def evaluate(model, eval_dataloader, device, rank=0, world_size=1):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(eval_dataloader, desc="Evaluating")
    else:
        pbar = eval_dataloader

    with torch.no_grad():
        for sample in pbar:
            labels = sample[0]   # string label
            imgs = sample[1].to(device) # [B, 3, H, W]
            points = sample[2].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
            text_ids = clip.tokenize(labels, truncate=True).to(device)

            text_features, image_features, lidar_features = model(text_ids, imgs, points)
            loss, _ = model.get_loss(text_features, image_features, lidar_features)

            total_loss += loss.item()
            num_batches += 1

    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        num_batches_tensor = torch.tensor(num_batches, device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        total_loss = total_loss_tensor.item()
        num_batches = num_batches_tensor.item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

###############################################################################
# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SparseCLIP model")
    parser.add_argument("--config_path", type=str, default="configs/sclip_168_20_knn_jitter.yaml", help="Path to the config file.")
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
    
    cfg["rank"] = rank
    cfg["world_size"] = world_size  

    ### Load models ###
    # Clip text encoder
    clip_model, clip_preprocess = clip.load(cfg["Model"]["clip_model"], jit=False)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    text_encoder = clip_model.encode_text
    img_encoder = clip_model.encode_image
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
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if world_size > 1:
        print(f"Model parameters: {model.module.param_count}")
    else:
        print(f"Model parameters: {model.param_count}")
    print(f"Lidar encoder parameters: {lidar_encoder.param_count}")

    ### Load dataset ###
    batch_size = cfg["Train"]["batch_size"]
    sparse_method = cfg["Dataset"]["sparse_method"]
    sparse_to_dense = load_sparse_method(sparse_method)

    train_dataset = Triplet_Object(
        jsonl_file=cfg["Dataset"]["train_data_path"],
        image_transform=image_transform,
        sparse_to_dense_fn=sparse_to_dense,
        image_tar_path=cfg["Dataset"].get("image_tar_path", None)
    )

    eval_dataset = Triplet_Object(
        jsonl_file=cfg["Dataset"]["eval_data_path"],
        image_transform=image_transform,
        sparse_to_dense_fn=sparse_to_dense,
        image_tar_path=cfg["Dataset"].get("image_tar_path", None)
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
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
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=cfg["Train"].get("num_workers", 0),
        pin_memory=True
    )

    ### Load configurations ###
    epochs = cfg["Train"]["epochs"]
    accumulation_steps = cfg["Train"]["accumulation_steps"]

    optimizer = optim.AdamW(
        list(model.lidar_encoder.parameters()) + list(model.projector.parameters()),
        lr=cfg["Train"]["learning_rate"],
        weight_decay=cfg["Train"]["weight_decay"]
    )
    steps_per_epoch = len(train_dataloader) // accumulation_steps
    total_training_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg["Train"]["warmup_ratio"] * total_training_steps)
    
    # Learning rate scheduler
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
        "tensorboard_logdir": cfg["Train"].get("tensorboard_logdir", None)
    }
    prepare_training()
    train(model, train_dataloader, eval_dataloader, optimizer, scheduler, train_config)
    