import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ctp(nn.Module):
    def __init__(self, text_encoder, image_encoder, lidar_encoder, loss_fn, alpha=0.5):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder      
        self.lidar_encoder = lidar_encoder
        self.projector = nn.Linear(1024, 512)
        self.alpha = alpha # weight for image and text features
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
        # cosine loss
        if self.loss_fn == 'cosine_similarity_loss':
            logits_eval, logits_per_text = self.cosine_similarity_loss(text_features, image_features, lidar_features)
            if logits_eval.shape[0] == logits_per_text.shape[0]:
                loss = self.combined_loss(logits_eval, logits_per_text)
        elif self.loss_fn == 'cosine_similarity_loss2':
            logits_eval, logits_per_text = self.cosine_similarity_loss2(text_features, image_features, lidar_features)
            if logits_eval.shape[0] == logits_per_text.shape[0]:
                loss = self.combined_loss(logits_eval, logits_per_text)
        elif self.loss_fn == 'TIL_cosine_matrix_loss':
            loss = self.TIL_cosine_matrix_loss(text_features, image_features, lidar_features)
        elif self.loss_fn == 'L_cosine_matrix_loss':
            loss = self.L_cosine_matrix_loss(text_features, image_features, lidar_features)
        elif self.loss_fn == 'cosine_matrix_loss_eval':
            logits_eval = self.cosine_matrix_loss_eval(text_features, image_features, lidar_features)
        # r2 loss
        elif self.loss_fn == 'r2_similarity_loss':
            loss, logits_eval = self.r2_similarity_loss(text_features, image_features, lidar_features)
        elif self.loss_fn == 'abs_cosine_matrix_loss_eval':
            logits_eval = self.abs_cosine_matrix_loss_eval(text_features, image_features, lidar_features)
        # l2 loss
        elif self.loss_fn == 'l2_similarity_loss':
            loss, logits_eval = self.l2_similarity_loss(text_features, image_features, lidar_features)
        elif self.loss_fn == 'l2_similarity_loss_no_mask':
            loss, logits_eval = self.l2_similarity_loss_no_mask(text_features, image_features, lidar_features)
        elif self.loss_fn == 'l2_similarity_loss_completed':
            loss, logits_eval = self.l2_similarity_loss_completed(text_features, image_features, lidar_features)
        elif self.loss_fn == 'l2_similarity_loss_completed_no_mask':
            loss, logits_eval = self.l2_similarity_loss_completed_no_mask(text_features, image_features, lidar_features)
        elif self.loss_fn == 'l2_similarity_loss_completed_stochastic':
            loss, logits_eval = self.l2_similarity_loss_completed_stochastic(text_features, image_features, lidar_features)
        elif self.loss_fn == 'cosine_cube_loss_completed':
            loss, logits_eval = self.cosine_cube_loss_completed(text_features, image_features, lidar_features)
        elif self.loss_fn == 'cosine_cube_loss_no_mask':
            loss, logits_eval = self.cosine_cube_loss_no_mask(text_features, image_features, lidar_features)
        else:
            raise Exception(f'Unknown loss function: {self.loss_fn}')
        return loss, logits_eval
    
    def cosine_similarity_loss(self, text_features, image_features, lidar_features):
        image_lidar_features = self.alpha * image_features + (1 - self.alpha) * lidar_features
        logit_scale = self.logit_scale.exp()
        logits_per_lidar = logit_scale * image_lidar_features @ text_features.t()
        logits_per_text = logits_per_lidar.t()
      
        return logits_per_lidar, logits_per_text

    def cosine_similarity_loss2(self, text_features, image_features, lidar_features):
        text_image_features = self.alpha * image_features + (1 - self.alpha) * text_features
        logit_scale = self.logit_scale.exp()
        logits_per_lidar = logit_scale * lidar_features @ text_image_features.t()
        logits_per_text = logits_per_lidar.t()
        
        return logits_per_lidar, logits_per_text

    def combined_loss(self, logits_per_lidar, logits_per_text):
        batch_size = logits_per_lidar.shape[0]
        labels = torch.arange(batch_size, device=logits_per_lidar.device)
        loss_lidar = F.cross_entropy(logits_per_lidar, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_lidar + loss_text) / 2        
        return loss        

    def TIL_cosine_matrix_loss(self, text_features, image_features, lidar_features):
        logit_scale = self.logit_scale.exp()
        logits_IT = logit_scale * image_features @ text_features.t()
        logits_TI = logits_IT.t()
        logits_LT = logit_scale * lidar_features @ text_features.t()
        logits_TL = logits_LT.t()
        logits_IL = logit_scale * image_features @ lidar_features.t()
        logits_LI = logits_IL.t()     

        loss_T_I = self.cos_matrix_loss(logits_IT, logits_TI)
        loss_L_T = self.cos_matrix_loss(logits_LT, logits_TL)
        loss_I_L = self.cos_matrix_loss(logits_IL, logits_LI)

        loss = (loss_T_I + loss_L_T + loss_I_L) / 3   
        return loss

    def L_cosine_matrix_loss(self, text_features, image_features, lidar_features):
        logit_scale = self.logit_scale.exp()
        logits_LT = logit_scale * lidar_features @ text_features.t()
        logits_TL = logits_LT.t()
        logits_IL = logit_scale * image_features @ lidar_features.t()
        logits_LI = logits_IL.t()     

        loss_L_T = self.cos_matrix_loss(logits_LT, logits_TL)
        loss_I_L = self.cos_matrix_loss(logits_IL, logits_LI)

        loss = (loss_L_T + loss_I_L) / 2   
        return loss
        
    def cos_matrix_loss(self, logits1, logits2):
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

        logits_eval = self.alpha * logits_IT + (1 - self.alpha) * logits_LT      
        return logits_eval

    def abs_cosine_matrix_loss_eval(self, text_features, image_features, lidar_features):
        logit_scale = self.logit_scale.exp()
        logits_IT = logit_scale * image_features @ text_features.t()
        logits_LT = logit_scale * lidar_features @ text_features.t()
        abs_logits_IT = torch.abs(logits_IT)
        abs_logits_LT = torch.abs(logits_LT)

        logits_eval = self.alpha * abs_logits_IT + (1 - self.alpha) * abs_logits_LT      
        return logits_eval
    
    def r2_similarity_loss(self, text_features, image_features, lidar_features):
        B_text, D = text_features.shape # x
        B_image, _ = image_features.shape # y
        B_lidar, _ = lidar_features.shape # z
        B_min = min(B_text, B_image, B_lidar)

        # Create meshgrid indices
        ti, tj, tk = torch.meshgrid(
            torch.arange(B_text),
            torch.arange(B_lidar),
            torch.arange(B_image),
            indexing='ij'
        )

        # Collect triplets: each (i,j,k) corresponds to 3 vectors
        t_triplet = text_features[ti].reshape(-1, D)   # [B_t*B_i*B_l, D]
        i_triplet = image_features[tj].reshape(-1, D)
        l_triplet = lidar_features[tk].reshape(-1, D)
        
        # Stack triplet: [B_t*B_i*B_l, 3, D]
        triplets = torch.stack([t_triplet, i_triplet, l_triplet], dim=1)
        # Compute Gram matrices for all triplets
        G = torch.bmm(triplets, triplets.transpose(1, 2))  # [N,3,3]
        # Eigen decomposition per sample
        eigvals = torch.linalg.eigvalsh(G)  # [N,3]
        # Compute R² = (λ_max-1) / (sum(λ)-1), range: [0, 1]
        r2 = (eigvals[:, -1] - 1) / (eigvals.sum(dim=1) - 1 + 1e-8)

        # Reshape to cube [B_text, B_lidar, B_image] (x, z, y)
        r2_cube = r2.reshape(B_text, B_lidar, B_image)
        logit_scale = self.logit_scale.exp()
        r2_cube_logits = logit_scale * r2_cube

        diag = torch.diagonal(r2_cube_logits, dim1=1, dim2=2)  # [B_text, B_min]

        # x-y plane loss
        xy_logits = self.plane_loss_masked(r2_cube_logits, B_min, diag, B_text, B_image, 'xy')
        # x-z plane loss
        xz_logits = self.plane_loss_masked(r2_cube_logits, B_min, diag, B_text, B_lidar, 'xz')  
        # z-y plane loss
        yz_logits = self.plane_loss_masked(r2_cube_logits, B_min, diag, B_lidar, B_image, 'yz')

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = (xy_loss + xz_loss + yz_loss) / 3

        logits_eval = r2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval

    def l2_similarity_loss(self, text_features, image_features, lidar_features):
        B_text, _ = text_features.shape
        B_image, _ = image_features.shape
        B_lidar, _ = lidar_features.shape
        B_min = min(B_text, B_image, B_lidar)

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

        # x-y plane loss
        l2_matrix_logits = self.plane_loss(l2_cube_logits, B_min, diag, B_text, B_image, 'xy')

        labels = torch.arange(B_text, device=l2_matrix_logits.device)
        l2_loss = F.cross_entropy(l2_matrix_logits, labels)

        logits_per_lidar = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_per_lidar

    def l2_similarity_loss_no_mask(self, text_features, image_features, lidar_features):
        B_text, dim = text_features.shape
        B_image, _ = image_features.shape
        B_lidar, _ = lidar_features.shape

        lidar_image_surface = torch.cdist(lidar_features, image_features, p=2) # z-y plane
        lidar_image_cube = lidar_image_surface.unsqueeze(0).expand(B_text,B_lidar,B_image)

        lidar_text_surface = torch.cdist(lidar_features, text_features, p=2) # x-z plane
        lidar_text_cube = lidar_text_surface.t().unsqueeze(-1).expand(B_text,B_lidar,B_image)

        text_image_surface = torch.cdist(text_features, image_features, p=2) # x-y plane
        text_image_cube = text_image_surface.unsqueeze(-2).expand(B_text,B_lidar,B_image)

        l2_cube = lidar_image_cube + lidar_text_cube + text_image_cube
        logit_scale = self.logit_scale.exp()
        l2_cube_logits = logit_scale * (1 - (l2_cube / (3 * np.sqrt(3)))) # range: [0, 3*sqrt(3)] -> [0, 1]

        # without mask related elements
        l2_surface_logits = []
        for idx in range(B_text):
            surface_xy = l2_cube_logits[:, idx, :] # select x-y plane
            surface_xy = surface_xy.clone()
            surface_xy[[0, idx]] = surface_xy[[idx, 0]] # swap first row and target row
            l2_surface = surface_xy.view(1, -1) # flatten
            l2_surface_logits.append(l2_surface)
            
        l2_matrix_logits = torch.cat(l2_surface_logits, dim=0)
        labels = torch.arange(B_text, device=l2_matrix_logits.device)
        l2_loss = F.cross_entropy(l2_matrix_logits, labels)

        logits_per_lidar = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_per_lidar

    def l2_similarity_loss_completed(self, text_features, image_features, lidar_features):
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

        # x-y plane loss
        xy_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_image, 'xy')
        # x-z plane loss
        xz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_lidar, 'xz')  
        # z-y plane loss
        yz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_lidar, B_image, 'yz')

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = (xy_loss + xz_loss + yz_loss) / 3

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval     

    def l2_similarity_loss_completed_no_mask(self, text_features, image_features, lidar_features):
        B_text, dim = text_features.shape # x
        B_image, _ = image_features.shape # y
        B_lidar, _ = lidar_features.shape # z
        B_min = min(B_text, B_image, B_lidar)

        # build l2 cube
        lidar_image_surface = torch.cdist(lidar_features, image_features, p=2) # z-y plane
        lidar_image_cube = lidar_image_surface.unsqueeze(0).expand(B_text,B_lidar,B_image)

        lidar_text_surface = torch.cdist(lidar_features, text_features, p=2) # x-z plane
        lidar_text_cube = lidar_text_surface.t().unsqueeze(-1).expand(B_text,B_lidar,B_image)

        text_image_surface = torch.cdist(text_features, image_features, p=2) # x-y plane
        text_image_cube = text_image_surface.unsqueeze(-2).expand(B_text,B_lidar,B_image)

        l2_cube = lidar_image_cube + lidar_text_cube + text_image_cube
        logit_scale = self.logit_scale.exp()
        l2_cube_logits = logit_scale * (1 - (l2_cube / (3 * np.sqrt(3)))) # range: [0, 3*sqrt(3)] -> [0, 1]

        # diag = torch.diagonal(l2_cube_logits, dim1=1, dim2=2)  # [B_text, B_min]

        # x-y plane loss
        xy_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xy')
        # x-z plane loss
        xz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xz')  
        # z-y plane loss
        yz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'yz')

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = (xy_loss + xz_loss + yz_loss) / 3

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval

    def l2_similarity_loss_completed_stochastic(self, text_features, image_features, lidar_features):
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

        # x-y plane loss
        xy_logits = self.plane_loss_stochastic(l2_cube_logits, B_min, diag, B_text, B_image, 'xy')
        # x-z plane loss
        xz_logits = self.plane_loss_stochastic(l2_cube_logits, B_min, diag, B_text, B_lidar, 'xz')  
        # z-y plane loss
        yz_logits = self.plane_loss_stochastic(l2_cube_logits, B_min, diag, B_lidar, B_image, 'yz')

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = (xy_loss + xz_loss + yz_loss) / 3

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval    
    
    def cosine_cube_loss_completed(self, text_features, image_features, lidar_features):
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
        l2_cube_logits = logit_scale * (1 - (l2_cube / (3 * np.sqrt(3)))) # range: [0, 3*sqrt(3)] -> [0, 1]

        diag = torch.diagonal(l2_cube_logits, dim1=1, dim2=2)  # [B_text, B_min]

        # x-y plane loss
        xy_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_image, 'xy')
        # x-z plane loss
        xz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_text, B_lidar, 'xz')  
        # z-y plane loss
        yz_logits = self.plane_loss_masked(l2_cube_logits, B_min, diag, B_lidar, B_image, 'yz')

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = (xy_loss + xz_loss + yz_loss) / 3

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval
    
    def cosine_cube_loss_no_mask(self, text_features, image_features, lidar_features):
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
        l2_cube_logits = logit_scale * (l2_cube / 3) # range: [-3, 3] -> [-1, 1]

        # diag = torch.diagonal(l2_cube_logits, dim1=1, dim2=2)  # [B_text, B_min]

        # x-y plane loss
        xy_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xy')
        # x-z plane loss
        xz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'xz')  
        # z-y plane loss
        yz_logits = self.plane_loss_no_mask(l2_cube_logits, B_min, 'yz')

        labels = torch.arange(B_min, device=xy_logits.device)

        xy_loss = F.cross_entropy(xy_logits, labels)
        xz_loss = F.cross_entropy(xz_logits, labels)
        yz_loss = F.cross_entropy(yz_logits, labels)

        l2_loss = (xy_loss + xz_loss + yz_loss) / 3

        logits_eval = l2_cube_logits[:, labels, labels].t()
        return l2_loss, logits_eval

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

    def plane_loss_stochastic(self, l2_cube_logits, B_min, diag, bach_1, bach_2, plane = 'xy'):
        num_to_sample = max(bach_1, bach_2) - 1
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
            l2_surface = surface.reshape(1, -1) # flatten, (1, N)

            target_value = l2_surface[:, idx : idx + 1] # save target value
            all_indices = torch.arange(l2_surface.numel(), device=l2_surface.device)
            available_indices = all_indices[all_indices != idx]
            sampled_indices = available_indices[
                torch.randperm(available_indices.numel(), device=l2_surface.device)[:num_to_sample]
            ]
            selected = l2_surface[:, sampled_indices]  # shape (1, num_to_sample)
            # add target value to taget indices
            combined = torch.cat(
                [selected[:, :idx], target_value, selected[:, idx:]],
                dim=1
            )  # shape (1, num_to_sample+1)
            l2_surface_logits.append(combined)
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
