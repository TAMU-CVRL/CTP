import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ctp(nn.Module):
    def __init__(self, text_encoder, image_encoder, lidar_encoder, loss_fn, alpha=1/3, beta=1/3, gamma=1/3, tau=1/2, masked=True):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder      
        self.lidar_encoder = lidar_encoder
        self.projector = nn.Linear(1024, 512)
        self.alpha = alpha # weight for image and text features
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.masked = masked
        self.loss_fn = loss_fn
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text_ids, image, lidar):
        lidar_features = self.lidar_encoder(lidar)
        lidar_features = self.projector(lidar_features) # project to match text feature dimension
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text_ids)
        text_features = text_features.float()
        
        image_features = image_features.to(lidar_features.dtype) # fp32
        text_features = text_features.to(lidar_features.dtype) # fp32

        # normalized features
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        lidar_features = F.normalize(lidar_features, dim=-1)
        
        return text_features, image_features, lidar_features

    def get_loss(self, text_features, image_features, lidar_features):
        loss = None
        logits_eval = None
        # matrix loss
        if self.loss_fn == 'cosine_matrix_loss':
            loss = self.cosine_matrix_loss(text_features, image_features, lidar_features)
        elif self.loss_fn == 'cosine_matrix_loss_eval':
            logits_eval = self.cosine_matrix_loss_eval(text_features, image_features, lidar_features)
        # tensor loss
        elif self.loss_fn == 'l2_tensor_loss':
            loss, logits_eval = self.l2_tensor_loss(text_features, image_features, lidar_features)
        elif self.loss_fn == 'cosine_tensor_loss':
            loss, logits_eval = self.cosine_tensor_loss(text_features, image_features, lidar_features)
        else:
            raise Exception(f'Unknown loss function: {self.loss_fn}')
        return loss, logits_eval   

    def cosine_matrix_loss(self, text_features, image_features, lidar_features):
        logit_scale = self.logit_scale.exp()
        logits_IT = logit_scale * ((image_features @ text_features.t()))
        logits_TI = logits_IT.t()
        logits_LT = logit_scale * ((lidar_features @ text_features.t()))
        logits_TL = logits_LT.t()
        logits_IL = logit_scale * ((image_features @ lidar_features.t()))
        logits_LI = logits_IL.t()     

        loss_T_I = self._cos_matrix_loss(logits_IT, logits_TI)
        loss_L_T = self._cos_matrix_loss(logits_LT, logits_TL)
        loss_I_L = self._cos_matrix_loss(logits_IL, logits_LI)

        loss = self.alpha * loss_T_I + self.beta * loss_L_T + self.gamma * loss_I_L
        return loss
        
    def _cos_matrix_loss(self, logits1, logits2):
        batch_size = logits1.shape[0]
        labels = torch.arange(batch_size, device=logits1.device)
        loss_1 = F.cross_entropy(logits1, labels)
        loss_2 = F.cross_entropy(logits2, labels)
        loss = (loss_1 + loss_2) / 2
        return loss

    def cosine_matrix_loss_eval(self, text_features, image_features, lidar_features):
        logit_scale = self.logit_scale.exp()
        logits_IT = logit_scale * image_features @ text_features.t()
        logits_LT = logit_scale * lidar_features @ text_features.t()

        logits_eval = self.tau * logits_IT + (1 - self.tau) * logits_LT      
        return logits_eval

    def l2_tensor_loss(self, text_features, image_features, lidar_features):
        B_text, dim = text_features.shape # x
        B_image, _ = image_features.shape # y
        B_lidar, _ = lidar_features.shape # z
        B_min = min(B_text, B_image, B_lidar)

        # build l2 cube, x: text, y: image, z: lidar
        lidar_image_surface = torch.cdist(lidar_features, image_features, p=2) # z-y plane
        lidar_image_cube = lidar_image_surface.unsqueeze(0).expand(B_text,B_lidar,B_image)

        lidar_text_surface = torch.cdist(lidar_features, text_features, p=2) # x-z plane
        lidar_text_cube = lidar_text_surface.t().unsqueeze(-1).expand(B_text,B_lidar,B_image)

        text_image_surface = torch.cdist(text_features, image_features, p=2) # x-y plane
        text_image_cube = text_image_surface.unsqueeze(-2).expand(B_text,B_lidar,B_image)

        l2_cube = lidar_image_cube + lidar_text_cube + text_image_cube
        logit_scale = self.logit_scale.exp()
        l2_cube_logits = logit_scale * (1 - (l2_cube / (3 * np.sqrt(3)))) # range: [0, 3*sqrt(3)] -> [0, 1]

        diag = torch.diagonal(l2_cube_logits, dim1=1, dim2=2)  # [B_text, B_min]

        if self.masked:
            xy_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_image, 'xy') # x-y plane loss
            xz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_lidar, 'xz') # x-z plane loss 
            yz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_lidar, B_image, 'yz') # z-y plane loss
        else:
            xy_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xy') # x-y plane loss
            xz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xz') # x-z plane loss
            yz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'yz') # z-y plane loss
                
        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = self.alpha * xy_loss + self.beta * xz_loss + self.gamma * yz_loss

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval
    
    def cosine_tensor_loss(self, text_features, image_features, lidar_features):
        B_text, dim = text_features.shape # x
        B_image, _ = image_features.shape # y
        B_lidar, _ = lidar_features.shape # z
        B_min = min(B_text, B_image, B_lidar)

        # build l2 cube
        lidar_image_surface = lidar_features@image_features.t() # z-y plane
        lidar_image_cube = lidar_image_surface.unsqueeze(0).expand(B_text,B_lidar,B_image)

        lidar_text_surface = lidar_features@text_features.t() # x-z plane
        lidar_text_cube = lidar_text_surface.t().unsqueeze(-1).expand(B_text,B_lidar,B_image)

        text_image_surface = text_features@image_features.t() # x-y plane
        text_image_cube = text_image_surface.unsqueeze(-2).expand(B_text,B_lidar,B_image)

        l2_cube = lidar_image_cube + lidar_text_cube + text_image_cube
        logit_scale = self.logit_scale.exp()
        l2_cube_logits = logit_scale * ((l2_cube + 1.5) / 4.5) # range: [0, 3*sqrt(3)] -> [0, 1]
        # l2_cube_logits = logit_scale * (2 * (l2_cube + 1.5) / 4.5 - 1) # range: [0, 3*sqrt(3)] -> [-1, 1]

        diag = torch.diagonal(l2_cube_logits, dim1=1, dim2=2)  # [B_text, B_min]

        if self.masked:
            xy_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_image, 'xy') # x-y plane loss
            xz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_lidar, 'xz') # x-z plane loss
            yz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_lidar, B_image, 'yz') # z-y plane loss
        else:
            xy_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xy') # x-y plane loss
            xz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xz') # x-z plane loss
            yz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'yz') # z-y plane loss

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        cosine_loss = self.alpha * xy_loss + self.beta * xz_loss + self.gamma * yz_loss

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return cosine_loss, logits_eval

    def plane_loss_masked(self, l2_cube_logits, B_min, diag, bach_1, bach_2, plane = 'xy'):
        mask = ~torch.eye(bach_1, bach_2, dtype=torch.bool)  # [bach_1, bach_2]
        l2_surface_logits = []
        for idx in range(B_min):
            # select plane
            if plane == 'xy':
                surface = l2_cube_logits[:, idx, :] # select x-y plane
            elif plane == 'xz':
                surface = l2_cube_logits[:, :, idx] # select x-z plane
            elif plane == 'yz':
                surface = l2_cube_logits[idx, :, :] # select y-z plane
            else:
                raise NotImplementedError
            # mask related elements
            masked = surface[mask[:, idx]][:, mask[idx, :]]
            flattened = masked.view(1, -1)
            diag_val = diag[idx, idx].view(1, 1)
            l2_surface = torch.cat([flattened[:, :idx], diag_val, flattened[:, idx:]], dim=1)
            l2_surface_logits.append(l2_surface)
        l2_matrix_logits = torch.cat(l2_surface_logits, dim=0)
        return l2_matrix_logits     
            
    def plane_loss_no_mask(self, l2_cube_logits, B_min, plane = 'xy'):
        # without mask related elements
        l2_surface_logits = []
        for idx in range(B_min):
            # select plane
            if plane == 'xy':
                surface = l2_cube_logits[:, idx, :] # select x-y plane
            elif plane == 'xz':
                surface = l2_cube_logits[:, :, idx] # select x-z plane
            elif plane == 'yz':
                surface = l2_cube_logits[idx, :, :] # select y-z plane
            else:
                raise NotImplementedError
            surface = surface.clone()
            surface[[0, idx]] = surface[[idx, 0]] # swap first row and target row
            l2_surface = surface.reshape(1, -1) # flatten
            l2_surface_logits.append(l2_surface)            
        l2_matrix_logits = torch.cat(l2_surface_logits, dim=0)
        return l2_matrix_logits     
        
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
