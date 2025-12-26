from copy import deepcopy
import torch
import torch.nn as nn
import clip
from lightly.loss.ntx_ent_loss import NTXentLoss

from .render import Renderer
from .selector import Selector
import numpy as np

clip_model, _ = clip.load("ViT-B/32", device='cpu')

class CLIP2Point(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.views = 10
        self.selector =  Selector(self.views)
        self.renderer = Renderer(points_radius=0.02)
        self.point_model = deepcopy(clip_model.visual)
        self.image_model = deepcopy(clip_model.visual)
        self.weights = nn.Parameter(torch.ones([]))
        self.criterion = NTXentLoss(temperature = 0.07)

        self.textual_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def infer(self, points, text_ids, rot=False):
        points = points - points.mean(dim=1, keepdim=True)                          # center at origin
        points = points / (points.abs().max(dim=1, keepdim=True).values + 1e-6)  # scale to [-1, 1]
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=rot)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        imgs = self.point_model(imgs)
        img_feats = imgs / imgs.norm(dim=-1, keepdim=True)

        text_feat = self.textual_encoder(text_ids)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True) 
        text_feat = text_feat.to(img_feats.dtype)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feat.t()
        return logits
    
    def forward(self, points, images, a, e, d):
        batch_size = points.shape[0]
        depths = self.renderer(points, a, e, d, 1, aug=True, rot=False)
        image_feat = self.image_model(images.squeeze(1)).detach()
        depth1 = depths[:, 0]
        depth2 = depths[:, 1]
        depths = torch.cat([depth1, depth2], dim=0)
        depths = self.point_model(depths)
        depth1_feat = depths[: batch_size]
        depth2_feat = depths[batch_size: ]
        depth_feat = (depth1_feat + depth2_feat) * 0.5
        depth_loss = self.criterion(depth1_feat, depth2_feat)
        image_loss = self.criterion(depth_feat, image_feat)
        return image_loss + depth_loss / (self.weights ** 2) + torch.log(self.weights + 1), image_loss, depth_loss
