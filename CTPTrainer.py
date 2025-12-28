import os
import argparse
import math
import clip
import wandb
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.ctp import ctp
from models.pointnet2 import pointnet2_encoder
from data.nuscenes_triplet import Triplet_Object_Nuscenes
from utils.img_utils import image_transform
from utils.pc_utils import load_sparse_method
from utils.model_utils import get_clip_encoders, prepare_training, load_config, build_scheduler, gather_features, pc_backbone

class CTPTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank = 0
        self.world_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._init_distributed()
        self.train_dataloader = self._build_dataloader()
        self.model = self._build_model()
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
        # initialize logging
        self.step = 0
        self.start_epoch = 0
        self._load_checkpoint_if_exists()
        self._init_logging()

    def _init_distributed(self):
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            self.device = f"cuda:{local_rank}"
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if self.rank == 0:
                print(f"[DDP] Initialized: rank {self.rank}/{self.world_size}")
        else:
            print(f"[Single-GPU] Training on {self.device}")

    def _build_dataloader(self):
        # TODO: can load evaluation dataset as well
        sparse_method = self.cfg["Dataset"]["sparse_method"]
        sparse_to_dense = load_sparse_method(sparse_method)
        
        tmpdir = os.getenv("TMPDIR", None)
        tar_path = self.cfg["Dataset"].get("image_tar_path", None)
        if tmpdir:
            file_name = os.path.basename(tar_path)
            local_tar_path = os.path.join(tmpdir, file_name)
            if os.path.exists(local_tar_path):
                    tar_path = local_tar_path
                    if self.rank == 0:
                        print(f"[INFO] Using local dataset at {tar_path}")

        dataset = Triplet_Object_Nuscenes(
            jsonl_file=self.cfg["Dataset"]["train_data_path"],
            image_transform=image_transform,
            sparse_to_dense_fn=sparse_to_dense,
            prompt = 'A '
        )

        sampler = DistributedSampler(dataset) if self.world_size > 1 else None
        return DataLoader(
            dataset,
            batch_size=self.cfg["Train"]["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.cfg["Train"].get("num_workers", 0),
            pin_memory=True,
            drop_last=True # drop last incomplete batch
        )

    def _build_model(self):
        pc_only = self.cfg["Model"]["pc_only"]
        clip_model_name = self.cfg["Model"]["clip_model"]
        pc_encoder_name = self.cfg["Model"]["point_model"].strip().lower()
        
        if pc_only:
            print(f"[INFO] Only Lidar encoder is trained.")
            clip_model, _ = clip.load(clip_model_name, jit=False, device=self.device)
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
            text_encoder, img_encoder = clip_model.encode_text, clip_model.encode_image
        else:
            print(f"[INFO] All encoders are trained.")
            text_encoder, img_encoder = get_clip_encoders(clip_model_name)
            text_encoder.to(self.device)
            img_encoder.to(self.device)

        pc_encoder = pc_backbone(pc_encoder_name, self.device)
        
        model = ctp(
            text_encoder=text_encoder,
            image_encoder=img_encoder,
            lidar_encoder=pc_encoder,
            loss_fn=self.cfg["Model"]["loss_fn"],
            alpha=self.cfg["Model"]["alpha"]
        ).to(self.device)

        if self.world_size > 1:
            model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)
        
        return model

    def _build_optimizer_and_scheduler(self):
        train_cfg = self.cfg["Train"]
        # train only lidar encoder or train all encoders
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = optim.AdamW(params, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
        
        steps_per_epoch = math.ceil(len(self.train_dataloader) / train_cfg["accumulation_steps"])
        total_steps = train_cfg["epochs"] * steps_per_epoch
        warmup_steps = int(train_cfg["warmup_ratio"] * total_steps)
        
        scheduler = build_scheduler(optimizer, train_cfg["scheduler"], warmup_steps, total_steps)
        return optimizer, scheduler

    def _init_logging(self):
            self.use_wandb = self.cfg["Train"].get("use_wandb", False)
            self.use_tb = self.cfg["Train"].get("use_tb", False)

            resume_id = self.cfg["Train"].get("wandb_id", None)
            resume_tb_dir = self.cfg["Train"].get("tb_log_dir", None)

            if self.rank == 0:
                if self.use_wandb:
                    print(f"[WandB] {'Resuming' if resume_id else 'Initializing'} WandB logging...")
                    wandb.init(
                        project="ctp", 
                        name=self.cfg["Name"], 
                        mode=self.cfg["Train"].get("wandb_mode", "offline"),
                        id=resume_id,
                        resume="allow"
                    )
                    if resume_id is None:
                        self.cfg["Train"]["wandb_id"] = wandb.run.id
                        print(f"[WandB] New Run started with ID: {wandb.run.id}")
                    wandb.config.update(self.cfg, allow_val_change=True)
                else:
                    print(f"[WandB] WandB logging is disabled.")
                
                if self.use_tb:
                    if resume_tb_dir and os.path.exists(resume_tb_dir):
                        log_dir = resume_tb_dir
                        print(f"[TensorBoard] Resuming from: {log_dir}")
                    else:
                        print(f"[TensorBoard] Initializing TensorBoard logging...")
                        log_dir = f"tb_logs/{datetime.now().strftime('%b%d_%H-%M-%S')}_{self.cfg['Name']}"
                    self.writer = SummaryWriter(log_dir=log_dir)
                else:
                    print(f"[TensorBoard] TensorBoard logging is disabled.")
                    self.writer = None
                
                os.makedirs(f"checkpoints/{self.cfg['Name']}", exist_ok=True)
            else:
                self.writer = None
                self.use_wandb = False

    def _load_checkpoint_if_exists(self):
        ckpt_path = self.cfg["Train"].get("checkpoint_path")
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model_to_load = self.model.module if self.world_size > 1 else self.model
            model_to_load.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.cfg["Train"]["wandb_id"] = ckpt.get('wandb_id', None)
            self.cfg["Train"]["tb_log_dir"] = ckpt.get('tb_log_dir', None)
            self.start_epoch = ckpt['epoch'] + 1
            self.step = ckpt['step']
            print(f"Loaded checkpoint from {ckpt_path}, starting from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch):
        if self.rank == 0:
            state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
            save_dir = Path(f"checkpoints/{self.cfg['Name']}")
            path = save_dir / f"ckpt_{epoch}.pt"

            checkpoint = {
                'epoch': epoch,
                'step': self.step,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'tb_log_dir': self.writer.get_logdir() if (self.use_tb and self.writer) else None,
                'wandb_id': wandb.run.id if (self.use_wandb and wandb.run) else None,
            }
            torch.save(checkpoint, path)

    def train(self):
        prepare_training()
        epochs = self.cfg["Train"]["epochs"]
        accumulation_steps = self.cfg["Train"]["accumulation_steps"]
        detailed_caption = self.cfg["Dataset"].get("detailed_caption", False)

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            updates_in_epoch = 0

            if self.world_size > 1:
                self.train_dataloader.sampler.set_epoch(epoch)

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}") if self.rank == 0 else self.train_dataloader

            for i, sample in enumerate(pbar):
                # load data
                caption = sample["caption"] if detailed_caption else sample["label"] # caption or annotation
                imgs = sample["image"].to(self.device) # [B, 3, H, W]
                points = sample["lidar"].permute(0, 2, 1).to(self.device) # [B, N, 3] -> [B, 3, N]
                text_ids = clip.tokenize(caption, truncate=True).to(self.device) # str -> tensor [B, L]

                # forward
                text_feats, img_feats, pc_feats = self.model(text_ids, imgs, points)
                text_feats, img_feats, pc_feats = gather_features(text_feats, img_feats, pc_feats, 
                                                                 gather_with_grad=True, world_size=self.world_size)
                
                core_model = self.model.module if self.world_size > 1 else self.model
                loss, _ = core_model.get_loss(text_feats, img_feats, pc_feats)
                
                # backward
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.step += 1
                    updates_in_epoch += 1
                    running_loss += loss.item()

                    if self.rank == 0:
                        pbar.set_description(f"Epoch {epoch} | Step {self.step} | Loss: {loss.item():.4f}")
                        if self.use_tb and self.writer:
                            self.writer.add_scalar("Loss/train_step", loss.item(), self.step)
                            self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], self.step)
                        if self.use_wandb:
                            wandb.log({"Loss/train_step": loss.item(), "LR": self.optimizer.param_groups[0]["lr"]}, step=self.step)

            # save checkpoint
            if self.world_size > 1: dist.barrier()
            
            if self.rank == 0:
                avg_loss = running_loss / max(updates_in_epoch, 1)
                if self.use_tb and self.writer:
                    self.writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
                if self.use_wandb:
                    wandb.log({"Loss/train_epoch": avg_loss, "Epoch": epoch}, step=self.step)
                print(f"--- Epoch {epoch} Done. Avg Loss: {avg_loss:.4f} ---")
                self.save_checkpoint(epoch)

        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ctp_default.yaml",
                        help="Path to the config file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    trainer = CTPTrainer(config)
    trainer.train()
