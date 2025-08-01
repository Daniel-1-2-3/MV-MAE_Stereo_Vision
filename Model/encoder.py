import torch
from torch import nn, Tensor
import numpy as np
from Model.sincos_pos_embeds import PosEmbed
from Model.vit import VitBlock
import random
class ViTMaskedEncoder(nn.Module):
    def __init__(self, 
            nviews=2,
            patch_size=8,
            embed_dim=768,
            in_channels=3,
            img_h_size=128,
            img_w_size=256, # Width of the fused image, with both views
            heads=16,
            depth=12
        ):
        
        super().__init__()
        self.nviews = nviews
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.heads = heads
        self.depth = depth
        
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

    def forward(self, x: Tensor):
        """
        Entire encoder feed forward operation

        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)

        Returns:
            x (Tensor):         Has shape (batch, unmasked_patches, embed_dim)
            mask (Tensor):      Has shape (batch, total_num_patches), where each vector in the 
                                last dimension is a binary mask with 0 representing unmasked, and 
                                1 representing masked
        """
        
        x = self.forward_early_conv(x) # Early conv to embed patches 
        # - (batch, total patches across both views, embed_dim)
        
        # Add sin/cos positional embeddings to each patch, addition element wise along last dim
        unmasked_x = self.add_pos_embeds(x) 
        
        masked_x, mask = self.random_view_masking(unmasked_x, mask_ratio=0.20)
        # - (batch, unmasked_patches, embed_dim)

        # Masked x passed through vit
        for block in self.vit_blocks:
            masked_x = block(masked_x)
            unmasked_x = block(unmasked_x)
        masked_x = self.norm_masked(masked_x)
        unmasked_x = self.norm_unmasked(unmasked_x)
        
        return masked_x, mask, unmasked_x,

    def random_view_masking(self, x: Tensor, mask_ratio=0.20):
        """
        The method masks the tensor, where either the left or right view is fully
        masked, while the other view is partially masked, with mask_ratio of the 
        patches masked. 
        
        Args:
            x (Tensor): Shape (batch, patches_left_view + patches_right_view, embed_dim)
            mask_ratio (float): Defaults to 0.20, ratio for partial masking.

        Returns:
            x_masked (Tensor):  Has shape (batch, num_unmasked_patches, embed_dim)
            mask (Tensor):      Has shape (batch, total_num_patches), where each vector in the 
                                last dimension is a binary mask with 0 representing unmasked, and 
                                1 representing masked
        """
        batch, num_patches, embed_dim = x.shape
        # x is currently (batch, patches_left_view + patches_right_view, embed_dim)
        
        x_kept = []
        mask_all = []
        num_mask = int(mask_ratio * (num_patches / 2))
        for i in range(batch):
            mask_view = random.random()
            mask = torch.ones(num_patches, dtype=torch.float32)
            
            # Address partial masking of views according to mask ratio
            if mask_view > 0.5: # Masking right view, partial masking for left view
                mask[:num_patches // 2].uniform_() # Fills left portion with random 
                # Argsort sorts the indices based on the values at each indice
                ids = torch.argsort(mask[:num_patches // 2], descending=False) # Sort ascending order
                keep_ids = ids[:num_patches // 2 - num_mask]
                # Assign 1 to all, and assign 0 to the unmasked patches
                mask[:num_patches // 2] = 1
                mask[keep_ids] = 0 
                
            else: # Masking left view, partial masking for right view
                mask[num_patches // 2:].uniform_()
                ids = torch.argsort(mask[num_patches // 2:], descending=False)
                keep_ids = ids[:num_patches // 2 - num_mask] + num_patches // 2 # Addition since this is the right side; end half of list
                mask[num_patches // 2:] = 1
                mask[keep_ids] = 0
            
            # Shape of mask = (num_patches,)
            keep_ids = torch.where(mask==0)[0] # Access list of indices where the value is 0 (unmasked)
            x_sample_i = x[i][keep_ids]
        
            x_kept.append(x_sample_i)
            mask_all.append(mask)

        x = torch.stack(x_kept)
        return x, torch.stack(mask_all)
    
    def add_pos_embeds(self, x: Tensor):
        # Shape of x: (batch, total_num_patches, embed_dim)
        # total_num_patches = nviews * grid_h_size * grid_w_size
        each_view_w = self.img_w_size // self.nviews
        each_view_h = self.img_h_size
        
        pos_embed = PosEmbed.get_2d_sincos_pos_embed(
            self.embed_dim, int(each_view_h // self.patch_size), int(each_view_w // self.patch_size)
        ) # (grid_h_size * grid_w_size, embed_dim)
        pos_embed = torch.from_numpy(pos_embed).to(x.device)
        pos_embed_all = pos_embed.repeat(self.nviews, 1) # (total_num_patches, embed_dim)
        
        x = x + pos_embed_all # Auto broadcasts over batch
        return x # (batch, total_num_patches, embed_dim)
    
    def construct_conv_layers(self):
        # Conv layers
        layers = []
        in_channels = self.in_channels
        nconvs = int(np.log2(self.patch_size)) 
        for i in range(nconvs):
            out_channels = self.embed_dim // (2 ** (nconvs - i + 1)) # The input channels of the next layer
            layers.append(nn.Conv2d( # In and out channels are dim 1
                in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1 # Output dims are halved 
            ))
            in_channels = out_channels
        layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(
            in_channels=in_channels, out_channels=self.embed_dim, kernel_size=1, stride=1
        )) # Final projection
        
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
     
if __name__ == "__main__":
    batch, height, width, nviews, in_channels = 32, 128, 128, 2, 3
    patch_size = 8
    embed_dim = 768
    width_total = nviews * width
    
    accurate_num_patches = (height * width) / (patch_size ** 2) * nviews
    print('Accurate num patches across both views', accurate_num_patches)
    
    x = torch.randn(batch, height, width_total, in_channels)
    print("(batch, height, fused_width, in_channels)", x.shape)
    
    encoder = ViTMaskedEncoder(nviews=nviews, patch_size=patch_size, 
        embed_dim=embed_dim, in_channels=in_channels)
    x, mask = encoder.forward(x)