import os
import yaml

from models.sllmclip import sclip_pair
from models.pointnet2 import pointnet2_encoder
from functions import TextLiDARPairDataset, collate_text_lidar_pairs, filter_small_pointclouds
from utils.processing import sparse_to_dense

from pathlib import Path
import torch

import clip
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from transformers import get_cosine_schedule_with_warmup

### Training ###
def train(model, train_dataloader, eval_dataloader, device, optimizer, scheduler, cfg):
    step = cfg.get("start_step", 0)
    start_epoch = cfg.get("start_epoch", 0)
    epochs = cfg.get("epochs", 10)
    device = cfg.get("device", "cuda")
    accumulation_steps = cfg.get("accumulation_steps", 1)
    tensorboard_logdir = cfg.get("tensorboard_logdir", None)

    if tensorboard_logdir is not None:
        train_writer = SummaryWriter(log_dir=tensorboard_logdir)
    else:
        train_writer = SummaryWriter()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
            for i, sample in enumerate(pbar):
                
                labels = sample[0]   # string label
                points = sample[1].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
                text_ids = clip.tokenize(labels, truncate=True).to(device)
                            
                optimizer.zero_grad()
                logits_per_lidar, logits_per_text = model(points, text_ids)
                loss = model.get_loss(logits_per_lidar, logits_per_text)
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()                

                step += 1
                running_loss += loss.item() * accumulation_steps
                
                # Log to TensorBoard
                train_writer.add_scalar("Loss/train_step", loss.item(), step)
                train_writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], step)
                
                if step % 2 == 0:
                    pbar.write(f"[LOG] Epoch {epoch} Step {step} Loss {loss.item():.4f}")

                # if step > 0 and step % 500 == 0:    
                #     model.eval()
                #     val_step_loss = evaluate(model, eval_dataloader, device)
                #     train_writer.add_scalar("Loss/val_step", val_step_loss, step)
                #     train_writer.add_scalars("Loss/epoch", {
                #         "train": loss.item(),
                #         "val": val_step_loss
                #     }, step)
                #     pbar.write(f"[EVAL] Step {step} | Validation Loss: {val_step_loss:.4f}")
                #     model.train()
        
        train_avg_loss = running_loss / len(train_dataloader)
        train_writer.add_scalar("Loss/train_epoch", train_avg_loss, epoch)

        model.eval()
        with torch.no_grad():
            val_avg_loss = evaluate(model, eval_dataloader, device)
        train_writer.add_scalar("Loss/val_epoch", val_avg_loss, epoch)
        
        train_writer.add_scalars("Loss/epoch", {
            "train": train_avg_loss,
            "val": val_avg_loss
        }, epoch)

        print(f"Epoch {epoch} | Train Loss: {train_avg_loss:.4f} | Validation Loss: {val_avg_loss:.4f}")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f"checkpoints/checkpoint_epoch{epoch}.pt")

def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        with tqdm(eval_dataloader, desc="Evaluating") as pbar:
            for sample in pbar:
                labels = sample[0]   # string label
                points = sample[1].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
                text_ids = clip.tokenize(labels, truncate=True).to(device)

                logits_per_lidar, logits_per_text = model(points, text_ids)
                loss = model.get_loss(logits_per_lidar, logits_per_text)

                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def prepare_training():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints") # save model checkpoints
    if not os.path.exists("runs"):
        os.makedirs("runs") # for TensorBoard logs

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

###############################################################################
# Main function
if __name__ == "__main__":
    cfg = load_config("configs/sclip.yaml")
    device = cfg["Train"]["device"] if torch.cuda.is_available() else "cpu"

    ### Load models ###
    # Clip text encoder
    clip_model, clip_preprocess = clip.load(cfg["Model"]["clip_model"], jit=False)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    text_encoder = clip_model.encode_text
    # PointNet++ encoder
    lidar_encoder = pointnet2_encoder.PointNet2Encoder() # [B, C, N] -> [B, C']
    lidar_encoder.to(device)
    # model
    model = sclip_pair(text_encoder=text_encoder, lidar_encoder=lidar_encoder)
    model.to(device)
    print(f"Model parameters: {model.param_count}")
    print(f"Lidar encoder parameters: {lidar_encoder.param_count}")

    ### Load dataset ###
    batch_size = cfg["Train"]["batch_size"]
    min_points = cfg["Dataset"]["min_points"]
    
    train_dataset = torch.load(cfg["Dataset"]["train_data_path"])
    filtered_train_dataset = filter_small_pointclouds(train_dataset, min_points=min_points)
    train_dataloader = torch.utils.data.DataLoader(
        filtered_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_text_lidar_pairs(batch, sparse_to_dense_fn=sparse_to_dense)
    )

    eval_dataset = torch.load(cfg["Dataset"]["eval_data_path"])
    filtered_eval_dataset = filter_small_pointclouds(eval_dataset, min_points=min_points)
    eval_dataloader = torch.utils.data.DataLoader(
        filtered_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_text_lidar_pairs(batch, sparse_to_dense_fn=sparse_to_dense)
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
    
    scheduler_cfg = cfg["Train"]["scheduler"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        num_cycles=scheduler_cfg["num_cycles"]
    )
    checkpoint_path = cfg["Train"].get("checkpoint_path", None)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
        "accumulation_steps": accumulation_steps,
        "start_epoch": start_epoch,
        "start_step": step,
        "epochs": epochs,
        "device": device,
        "tensorboard_logdir": cfg["Train"].get("tensorboard_logdir", None)
    }
    prepare_training()
    train(model, train_dataloader, eval_dataloader, device, optimizer, scheduler, train_config)
    