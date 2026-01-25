import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionEncoder(nn.Module):
    def __init__(self, pc_encoder, num_regions):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.num_regions = num_regions

    def forward(self, points):
        B, _, N = points.shape
        split_points = torch.chunk(points, self.num_regions, dim=2)

        region_feats = []
        for p in split_points:
            feat = self.pc_encoder(p)
            region_feats.append(feat)

        region_feats = torch.stack(region_feats, dim=1)
        return region_feats
