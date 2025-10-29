import torch
from torch import nn, Tensor
import numpy as np
from MAE_Model.sincos_pos_embeds import PosEmbed
from MAE_Model.vit import VitBlock
import random
class ViTMaskedEncoder(nn.Module):
    def __init__(self, 
            nviews: int = 2,
            patch_size: int = 6,
            embed_dim: int = 768,
            in_channels: int = 3,
            img_h_size: int = 84,
            img_w_fused_size: int = 168, # Width of the fused image, with both views
            heads: int = 8,
            depth: int = 6,
            masking_ratio: float = 0.75,
        ):
        
        super().__init__()
        self.nviews = nviews
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_fused_size = img_w_fused_size
        self.heads = heads
        self.depth = depth
        self.masking_ratio = masking_ratio
        
        # Positional embeddings
        with torch.no_grad():
            each_view_w = img_w_fused_size // nviews
            each_view_h = img_h_size
            pe_np = PosEmbed.get_2d_sincos_pos_embed(
                embed_dim, int(each_view_h // patch_size), int(each_view_w // patch_size)
            )  # (grid_h * grid_w, embed_dim)
            pe = torch.from_numpy(pe_np).repeat(nviews, 1)  # (total_patches, embed_dim)
        self.register_buffer("pos_embed_all", pe, persistent=False)
        
        self.forward_conv = self.construct_conv_layers()
        self.vit_blocks = nn.ModuleList([
            VitBlock(
                embed_dim=self.embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                attn_drop_rate=0.1,
                mlp_drop_rate=0.1,
                path_drop_rate=0.05
            ) for _ in range(self.depth)
        ])
        self.norm_masked = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)
        self.norm_unmasked = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)

    def forward(self, x: Tensor, mask_x: bool):
        """
        Entire encoder feed forward operation

        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)
            mask_x: Whether to mask 75% of x. Training pipeline masks when training mvmae,
                        but does not mask when using the encoder output as input.

        Returns:
            x (Tensor): Either has shape (batch, unmasked_patches, embed_dim) or
                        shape (batch, total_patches, embed_dim)
            mask (Tensor):      Has shape (batch, total_num_patches), where each vector in the 
                                last dimension is a binary mask with 0 representing unmasked, and 
                                1 representing masked
        """
        
        x = self.forward_early_conv(x) # Early conv to embed patches 
        # - (batch, total patches across both views, embed_dim)
        
        # Add sin/cos positional embeddings to each patch, addition element wise along last dim
        x = self.add_pos_embeds(x)
        mask = None
        if mask_x:
            x, mask = self.random_view_masking(x) # (batch, unmasked_patches, embed_dim)

        for block in self.vit_blocks:
            x = block(x)
        
        x = self.norm_masked(x) if mask_x else self.norm_unmasked(x)
        return x, mask

    def random_view_masking(self, x: torch.Tensor):
        """
        The method masks the tensor, where either the left or right view is fully
        masked, while the other view is partially masked, with mask_ratio of the 
        patches masked. 

        Args:
            x (Tensor): Shape (batch, patches_left_view + patches_right_view, embed_dim)

        Returns:
            x_masked (Tensor):  Has shape (batch, num_unmasked_patches, embed_dim)
            mask (Tensor):      Has shape (batch, total_num_patches), where each vector in the 
                                last dimension is a binary mask with 0 representing unmasked, and 
                                1 representing masked
        """
        device = x.device
        batch, num_patches, embed_dim = x.shape
        half = num_patches // 2

        # How many to mask on the partially-masked side
        mask_ratio = (self.masking_ratio - 0.5) / 0.5
        num_mask = int(mask_ratio * half)
        keep_per_half = half - num_mask  # Fixed number of kept patches per sample

        mask_right = torch.rand(batch, device=device) > 0.5 # Choose side to fully mask
        scores = torch.rand(batch, half, device=device) # Random scores for each patch on 1 half, low scores get kept
        keep_left  = torch.topk(-scores, keep_per_half, dim=1).indices # Indices to keep on partially masked side
        keep_right = torch.topk(-scores, keep_per_half, dim=1).indices + half

        # If fully masking right side, then use keep_left, is not fully masking right side, then keep_right
        keep_idx = torch.where(mask_right.unsqueeze(1), keep_left, keep_right) # (batch, kept_indices)
        mask = torch.ones(batch, num_patches, device=device, dtype=torch.float32)
        row = torch.arange(batch, device=device).unsqueeze(1)
        mask[row, keep_idx] = 0.0

        gather_idx = keep_idx.unsqueeze(-1).expand(batch, keep_per_half, embed_dim)   # (B, K, D)
        x_masked = torch.gather(x, dim=1, index=gather_idx)                           # (B, K, D)

        return x_masked, mask
    
    def add_pos_embeds(self, x: Tensor):
        return x + self.pos_embed_all.to(dtype=x.dtype)
    
    def construct_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        # Instead of requiring patch_size to be a power of 2, compute stride factors that multiply to patch_size
        stride_factors = []
        
        n = self.patch_size
        while n > 1:
            if n % 2 == 0:
                stride_factors.append(2)
                n //= 2
            else:
                stride_factors.append(n)
                break

        for i, s in enumerate(stride_factors):
            out_channels = max(self.embed_dim // (2 ** (len(stride_factors) - i)), 32)  
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=max(3, s),
                stride=s,
                padding=max(3, s) // 2
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Final projection to embed_dim
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.embed_dim,
            kernel_size=1,
            stride=1
        ))

        forward_conv = nn.Sequential(*layers)
        return forward_conv
        
    def forward_early_conv(self, x: Tensor):
        """
        Extract features with several convolutional layers before being
        fed into the transformer
        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)
        Returns:
            x (Tensor): After being passed through forward conv layers and a ReLU, 
                        x now has a shape of (batch, total_num_patches, embed_dim)
        """
        batch, height, width_total, channels = x.shape

        # Temporarily convert (b, h, w, c) into (b * nviews, h, w_per_cam, c)
        width_per_view = width_total // self.nviews
        x = torch.split(x, width_per_view, dim=2) # Tuple of tensors, each (b, h, w_per_cam, c)
        x = torch.cat(x, dim=0) # (b * nviews, h, w_per_cam, c)
        x = x.permute(0, 3, 1, 2)
        
        x = self.forward_conv(x) # Shape (b * nviews, embed_dim, reduced_height, reduced_width)
        
        # Reshape x to be fed into the transformer
        x = x.permute(0, 2, 3, 1) # To (b * nviews, reduced_height, reduced_width, embed_dim)
        x = x.reshape(x.shape[0], -1, x.shape[-1]) # To (b * nviews, h * w = num_patches, embed_dim)
        x = torch.chunk(x, self.nviews, dim=0)
        x = torch.cat(x, dim=1)
        # total_num_patches = num_patches * nviews
        
        return x # (batch, total_num_patches, embed_dim)
