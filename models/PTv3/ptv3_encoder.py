import torch
import torch.nn as nn
from .model import PointTransformerV3
import torch_scatter

class PTv3Encoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=1024, 
                 grid_size=0.02,
                 enc_channels=(32, 64, 128, 256, 512), 
                 enc_depths=(2, 2, 2, 6, 2)):
        super().__init__()
        self.grid_size = grid_size
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            enc_channels=enc_channels,
            enc_depths=enc_depths,
            cls_mode=True,
            enable_flash=True,
        )
        last_enc_channel = enc_channels[-1]
        self.fc = nn.Linear(last_enc_channel, out_channels)

    def forward(self, xyz):
        """
        xyz: [B, C, N]
        """
        B, C, N = xyz.shape
        device = xyz.device
        mask = (xyz[:, :3, :] != 0).any(dim=1)  # [B, N], remove zero points
        
        coords = xyz[:, :3, :].transpose(1, 2)[mask] # [num_valid_points, 3]
        feats = xyz.transpose(1, 2)[mask] # [num_valid_points, C]
        batch = torch.arange(B, device=device).unsqueeze(1).repeat(1, N)
        batch = batch[mask]  # [num_valid_points]
        
        # Build the data dict for PTv3
        data_dict = {
            "coord": coords,
            "feat": feats,
            "batch": batch,
            "grid_size": self.grid_size
        }
        
        point = self.backbone(data_dict)
        
        # Global Max Pooling
        global_feat = torch_scatter.scatter_max(point.feat, point.batch, dim=0)[0]
        
        # [B, 1024]
        out = self.fc(global_feat)
        return out
