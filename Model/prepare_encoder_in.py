import torch
from torch import nn
from torch import Tensor
import math, random

class PatchEmbedding(nn.Module):
    def __init__ (self, in_channels, img_size, patch_size, embed_dim, training=True):
        """
        Each view is passed through a series of 4 convolutional layers (with shared weights across views) 
        to extract patch embeddings. Fixed sine/cosine positional embeddings are then added to each patch 
        to encode spatial information. View embeddings are also added. 

        Args:
            in_channels (int):  Typically 3 for RGB input
            img_size (int):     Side length of the input image
            patch_size (int):   Side length of each patch
            embed_dim (int):    Dimension of each patch token vector
            training (bool):    During training, one view is masked, and 25% of other 
                                view's patches are also masked. (default True)
        """
        super().__init__()
        
        self.training = training
        total_patches = int((img_size / patch_size) ** 2)
        self.patch_extract = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim//4, embed_dim//4*2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim//4*2, embed_dim//4*3, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim//4*3, embed_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((int(math.sqrt(total_patches)), int(math.sqrt(total_patches))))
        ) # (batch, embed_dim, sqrt(total_patches), sqrt(total_patches))
        self.positional_embeds = self.sin_cos_embed(int(math.sqrt(total_patches)), embed_dim)
        self.view1_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        self.view2_embed = nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=False)
        self.masked_ids = None
    
    def sin_cos_embed(self, grid_size, embed_dim):
        """
        Generate sin/cos embeddings representing the 
        position of each patch token in the image (grid).

        Args:
            grid_size (int):    The number of patches determines the amount 
                                of positional embeddings that are needed
            embed_dim (int):    The dimension of each patch token determines 
                                the length of its positional embedding (how many individual 
                                sine and cosine components are used to encode its position.
        Returns:
            (Tensor):   Outputs a shape of (num_patches, embed_dim), 
                        able to be directly concatenated to the patch embedding
        """
        # Grid of patch positions
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0) # (2, grid_size, grid_size)
        grid = grid.flatten(2, -1)
        grid = grid.transpose(0, 1) # (total_patches, 2)

        # Apply sin/cos to x and y poses
        dim_half = embed_dim // 2
        term = torch.exp(torch.arange(0, dim_half, 2) * (-math.log(10000.0) / dim_half))
        pos_x = grid[:, 0].unsqueeze(1) * term # (num_patches, dim_half//2)
        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=1)  # (num_patches, dim_half)
        pos_y = grid[:, 1].unsqueeze(1) * term
        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=1)

        # Concat x and y embeddings
        pos_embed = torch.cat([pos_x, pos_y], dim=1) # (num_patches, embed_dim)
        return pos_embed

    def random_mask(self, x, mask_ratio):
        batch, total_patches, embed_dim = x.shape
        num_keep = int(total_patches * (1 - mask_ratio))

        scores = torch.rand(batch, total_patches, device=x.device) # Random score for each token
        ids_sorted = torch.argsort(scores, dim=1)
        ids_keep = ids_sorted[:, :num_keep]
        self.masked_ids = ids_keep

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim)) # (batch, kept_patches, embed_dim)
        return x_masked

    def forward(self, x1: Tensor, x2: Tensor):
        """
        Args:
            x1 (Tensor):    Tensor of first view in stereo vision (batch, channels, H, W)
            x2 (Tensor):    Tensor of second view in stereo vision (batch, channels, H, W)

        Returns:
            (Tensor):   Output is a tensor that includes all unmasked patches, with a
                        shape of (batch, num_unmasked_patches, embed_dim)
        """
        x1 = self.patch_extract(x1)
        x1 = x1.flatten(2, 3)
        x1 = x1.transpose(1, 2) # (batch, total_patches, embed_dim)
        x1 += self.positional_embeds.unsqueeze(0) + self.view1_embed
        
        x2 = self.patch_extract(x2)
        x2 = x2.flatten(2, 3)
        x2 = x2.transpose(1, 2)
        x2 += self.positional_embeds.unsqueeze(0) + self.view2_embed
        
        x = torch.cat((x1, x2), dim=1)
        if self.training:
            partial_mask = random.choice([x1, x2]) # Other is fully masked
            x = self.random_mask(partial_mask, 0.25)

        return x