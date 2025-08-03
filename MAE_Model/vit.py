from torch import nn
from timm.layers import DropPath

class VitBlock(nn.Module):
    def __init__(self, 
            embed_dim: int,
            heads: int,
            mlp_ratio: float,
            attn_drop_rate: float,
            mlp_drop_rate: float,
            path_drop_rate: float, # Drops entire connection
        ):
        
        super().__init__()
        self.drop_path = DropPath(drop_prob=path_drop_rate)
        
        self.multi_head_self_attn= nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=heads, 
            dropout=attn_drop_rate, 
            batch_first=True
        )
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_drop_rate),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(p=mlp_drop_rate),
        )
        
        # Norms - "layer_norm_eps_1e-6", eps is small constant to denominator to prevent devision by 0
        self.norm_before_attn = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.norm_before_mlp = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        
    def forward(self, x):
        """
        ViT block includes multi headed self attention and multi-layer perception (mlp) layer,
        each with prenorm and residual/skip connection.

        Args:
            x (Tensor): Shape (batch, num_patches, embed_dim)

        Returns:
            x (Tensor): Same shape as input (batch, num_patches, embed_dim)
        """
        residual = x
        x = self.norm_before_attn(x)
        x, attn_weights = self.multi_head_self_attn(x, x, x)
        x = self.drop_path(x)
        x = x + residual
        
        residual = x
        x = self.norm_before_mlp(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        
        return x