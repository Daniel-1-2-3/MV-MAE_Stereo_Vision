import torch
from torch import nn, Tensor
from Model.vit import VitBlock

class ViTMaskedDecoder(nn.Module):
    def __init__(self,
            nviews=2,
            patch_size=8,
            encoder_embed_dim=768,
            img_h_size=128,
            img_w_size=128,
            decoder_embed_dim=512,
            in_channels=3,
            heads=16,
            depth=8
        ):
        
        super().__init__()
        self.nviews = nviews
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.in_channels = in_channels
        self.heads = heads
        self.depth = depth
        self.num_total_patches = int((img_h_size * img_w_size) // (patch_size ** 2) * nviews)
        
        self.proj = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.mask_tokens = nn.Parameter(torch.randn(1, 1, self.decoder_embed_dim))
        # Learnable, since attention in the encoder has already mixed spatial locations
        self.pos_embeds = nn.Parameter(torch.randn(1, self.num_total_patches, self.decoder_embed_dim)) 
        
        self.vit_blocks = nn.ModuleList([
            VitBlock(
                embed_dim=self.decoder_embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                attn_drop_rate=0.1,
                mlp_drop_rate=0.1,
                path_drop_rate=0.05
            ) for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(normalized_shape=self.decoder_embed_dim, eps=1e-6)
        
    def forward(self, x: Tensor, mask: Tensor):
        """
        The decoder uses a linear projection to compress the embedding dimension.
        Then, [MASK] tokens are inserted 

        Args:
            x (Tensor):         Has shape (batch, unmasked_patches, embed_dim)
            mask (Tensor):      Has shape (batch, total_num_patches), where each vector in the 
                                last dimension is a binary mask with 0 representing unmasked, and 
                                1 representing masked

        Returns:
            x (Tensor): (batch, total number of patches, decoder embed dim)
        """
        x = self.proj(x)
        
        batch, _, dim = x.shape
        total_patches = mask.shape[1] 
        unmasked_indices = (mask == 0)
        x_full = self.mask_tokens.expand(batch, total_patches, dim).clone() # [MASK] tokens for all patches
        x_full[unmasked_indices] = x.reshape(-1, dim) # Scatter the encoder outputs into the unmasked positions
        x = x_full + self.pos_embeds
        
        for block in self.vit_blocks:
            x = block(x)
        x = self.norm(x)

        return x