import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from clip import clip

from .mv_utils_zs2 import Realistic_Projection

class PointCLIPV2(nn.Module):
    def __init__(self, text_encoder, image_encoder, num_views=10, channel=512, device='cuda'):
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
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        
        self.view_weights = torch.ones(self.num_views, dtype=torch.float32, device=self.device)

    def real_proj(self, pc, imsize=224):
        img = self.get_img(pc).cuda()
        img = F.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img
    
    def model_inference(self, pc, text_ids):
        with torch.no_grad():
            pc = pc - pc.mean(dim=1, keepdim=True)
            scale = pc.abs().max(dim=1, keepdim=True)[0]
            pc = pc / scale  # normalize to unit cube
            # Realistic Projection
            images = self.real_proj(pc)            
            images = images.type(self.dtype)
            
            # Image features
            image_feat = self.visual_encoder(images)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            
            image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
            image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
                        
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)

            # Text features
            text_feat = self.textual_encoder(text_ids)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  
            text_feat = text_feat.repeat(1, self.num_views)  # [T, V*C]
            text_feat = text_feat.to(image_feat_w.dtype)

            logits = 100. * image_feat_w @ text_feat.t()
        return logits