from torch import nn, Tensor
import torch
import einops
import torch.nn.functional as F

from MAE_Model.encoder import ViTMaskedEncoder
from MAE_Model.decoder import ViTMaskedDecoder
import matplotlib.pyplot as plt

class MAEModel(nn.Module):
    def __init__(self, 
            nviews=2,
            patch_size=8,
            encoder_embed_dim=768,
            decoder_embed_dim=512,
            encoder_heads=16,
            decoder_heads=16,
            in_channels=3,
            img_h_size=128,
            img_w_size=128, 
        ):
        super().__init__()
        self.nviews = nviews
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.img_w_fused = self.nviews * self.img_w_size
        self.num_patches = int((self.img_h_size * self.img_w_size) // (patch_size ** 2) * nviews)
        
        self.encoder = ViTMaskedEncoder(depth=4, embed_dim=self.encoder_embed_dim)
        self.decoder = ViTMaskedDecoder(depth=2, encoder_embed_dim=self.encoder_embed_dim, decoder_embed_dim=self.decoder_embed_dim)
        self.out_proj = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * in_channels)
    
    def forward(self, x: Tensor):
        """
        Whole pipeline of the MV-MAE model: patchified, 
        then passed through encoder, mask tokens added, and 
        passed through decoder. 

        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)

        Returns:
            out (Tensor):   (batch, total_patches, patch_size^2 * channels) Input is masked, then
                                    fed through the encoder, then the decoder
            mask (Tensor):      Has shape (batch, total_num_patches), where each vector in the 
                                last dimension is a binary mask with 0 representing unmasked, and 
                                1 representing masked
            z (Tensor):    (batch, total_patches, patch_size^2 * channels) This is the input to the 
                                    actor in the pipeline. It is the input, without masking, passed through the encoder
        """
        masked_x, mask, z = self.encoder(x)
        out = self.decoder(masked_x, mask)
        out = self.out_proj(out)
        return out, mask, z
    
    def compute_loss(self, out: Tensor, truth: Tensor, mask: Tensor):
        """
        Compute MSE loss

        Args:
            out (Tensor): (batch, total_patches, patch_size^2 * channels)
            truth (Tensor): Fused views (batch, height, width_total, channels)
            mask (Tensor):  Has shape (batch, total_num_patches), where each vector in the 
                            last dimension is a binary mask with 0 representing unmasked, and 
                            1 representing masked

        Returns:
            loss (Tensor): MSE loss, retains grad information
        """
        mask = mask.to(out.device)
        truth = self.patchify(truth).to(out.device)
        loss_per_patch = F.mse_loss(out, truth, reduction='none').mean(dim=-1) # Reduction 'None' for per-element loss
        loss = (loss_per_patch * mask).sum() / mask.sum() # Only calculate loss for masked patches
        return loss
    
    def render_reconstruction(self, x: Tensor):
        """
        Args:
            x (Tensor): (batch, total_patches, patch_size^2 * channels)
        """
        x = self.unpatchify(x)[0]
        mean = torch.tensor([0.51905, 0.47986, 0.48809], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.17454, 0.20183, 0.19598], device=x.device).view(3, 1, 1)
        x = x * std + mean
        x = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imshow(x)
        plt.show()
        
    def patchify(self, x: Tensor):
        """
        Convert the ground truth views into patches to match the format of the
        decoder output, in order to compute loss

        Args:
            x (Tensor): Representing all the views stitched together horizontally,
                        with a shape of (batch, height, width_total, channels)
        
        Returns:
            x (Tensor): (batch, total_patches, patch_size^2 * channels)
        """
        batch, height, w_total, in_channels = x.shape
        assert w_total % self.nviews == 0, "Width must be divisible by number of views"

        # Split along width into views into (b, h, w, c) x nviews
        views = torch.chunk(x, self.nviews, dim=2)  # List of [b, h, w, c]
        patchified_views = [ # Rearrange
            einops.rearrange(v, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', 
                p1=self.patch_size, p2=self.patch_size) for v in views
        ]

        return torch.cat(patchified_views, dim=1)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, total_patches, patch_size**2 * C)
        Returns:
            imgs: (B, C, H, W_fused) with views stitched horizontally
        """
        _, total_patches, dim = x.shape
        c = dim // (self.patch_size ** 2)

        patches_per_view = total_patches // self.nviews
        h = self.img_h_size // self.patch_size
        w = self.img_w_size // self.patch_size

        views = torch.split(x, patches_per_view, dim=1) # Split the concatenated views

        imgs = [
            einops.rearrange(v, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=c)
            for v in views
        ]
        
        return torch.cat(imgs, dim=3)
