import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class sllm_clip(nn.Module):
    def __init__(self, text_encoder, lidar_encoder):
        super().__init__()
        self.text_encoder = text_encoder        
        self.lidar_encoder = lidar_encoder
        self.projector = nn.Linear(1024, 512)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, lidar, text_ids):
        lidar_features = self.lidar_encoder(lidar)
        lidar_features = self.projector(lidar_features) # project to match text feature dimension
        text_features = self.text_encoder(text_ids)
        text_features = text_features.float()
        
        # normalized features
        lidar_features = lidar_features / lidar_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_lidar = logit_scale * lidar_features @ text_features.t()
        logits_per_text = logits_per_lidar.t()
        
        return logits_per_lidar, logits_per_text

    def get_loss(self, logits_per_lidar, logits_per_text):
        batch_size = logits_per_lidar.shape[0]
        labels = torch.arange(batch_size, device=logits_per_lidar.device)
        loss_lidar = F.cross_entropy(logits_per_lidar, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        return (loss_lidar + loss_text) / 2
    
    @property
    def param_count(self):
        n = sum(p.numel() for p in self.parameters())
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return str(n)
