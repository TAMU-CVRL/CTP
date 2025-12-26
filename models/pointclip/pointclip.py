import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mv_utils_zs import PCViews

class PointCLIP(nn.Module):
    def __init__(self, text_encoder, image_encoder, num_views=6, channel=512, device='cuda'):
        super().__init__()
        # Encoders from CLIP
        self.visual_encoder = image_encoder
        self.textual_encoder = text_encoder

        # Logit scale (CLIP temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Multi-view configuration
        self.num_views = num_views
        self.channel = channel
        self.device = device
        self.dtype = torch.float32

        # Multi-view projection utility
        self.pc_views = PCViews()
        self.get_img = self.pc_views.get_img

        # Storage for postprocessing
        self.feat_store = []
        self.label_store = []

    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img
    
    def model_inference(self, pc, text_ids):
        with torch.no_grad():
            pc = pc - pc.mean(dim=1, keepdim=True)
            scale = pc.abs().max(dim=1, keepdim=True)[0]
            pc = pc / scale  # normalize to unit cube
            # Project to multi-view depth maps
            images = self.mv_proj(pc).type(self.dtype)

            # Image features
            image_feat = self.visual_encoder(images)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) 
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)

            # Text features
            text_feat = self.textual_encoder(text_ids)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  
            text_feat = text_feat.repeat(1, self.num_views)  # [T, V*C]

            # Classification logits
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_feat @ text_feat.t() * 1.0
        
        return logits
