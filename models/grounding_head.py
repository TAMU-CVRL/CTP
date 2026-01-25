import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GroundingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text_features, region_features, region_mask=None):
        text_features = F.normalize(text_features, dim=-1)
        region_features = F.normalize(region_features, dim=-1)

        logits = torch.einsum("bd,bkd->bk", text_features, region_features)
        logits = self.logit_scale.exp() * logits

        if region_mask is not None:
            logits = logits.masked_fill(~region_mask, -1e9)

        return logits
