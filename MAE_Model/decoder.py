import functools
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from MAE_Model.vit import VitBlock

class ViTMaskedDecoder(nn.Module):
    nviews: int = 2
    patch_size: int = 6
    encoder_embed_dim: int = 768
    img_h_size: int = 84
    img_w_size: int = 84
    decoder_embed_dim: int = 512
    in_channels: int = 3
    heads: int = 8
    depth: int = 4

    def setup(self):
        # Total number of patches across all views
        self.num_total_patches = int(
            (self.img_h_size * self.img_w_size) // (self.patch_size**2) * self.nviews
        )

        # Project encoder dim -> decoder dim
        self.proj = nn.Dense(features=self.decoder_embed_dim)

        # Learnable [MASK] token (broadcasted across all masked patch positions)
        self.mask_tokens = self.param(
            "mask_tokens",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.decoder_embed_dim),
        )

        # Learnable positional embeddings (decoder typically uses learnable positions)
        self.pos_embeds = self.param(
            "pos_embeds",
            nn.initializers.normal(stddev=1.0),
            (1, self.num_total_patches, self.decoder_embed_dim),
        )

        # Transformer blocks
        self.vit_blocks = [
            VitBlock(
                embed_dim=self.decoder_embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                attn_drop_rate=0.1,
                mlp_drop_rate=0.1,
                path_drop_rate=0.05,
            )
            for _ in range(self.depth)
        ]

        self.norm = nn.LayerNorm(epsilon=1e-6)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """
        Args:
            x:    (batch, unmasked_patches, encoder_embed_dim)  -- from encoder (masked case)
                  or (batch, total_patches, encoder_embed_dim) -- if you choose to pass unmasked all
            mask: (batch, total_patches) where 0=keep, 1=mask

        Returns:
            x: (batch, total_patches, decoder_embed_dim)
        """
        # Project to decoder dim
        x = self.proj(x)  # (B, K, D_dec)

        batch, k_patches, d_dec = x.shape
        total_patches = mask.shape[1]

        # If already full-length, RETURN A FULL-LENGTH TENSOR (same shape as scatter branch)
        # This avoids lax.cond branch shape mismatch.
        def _already_full():
            unmasked = (mask == 0)
            x_full = jnp.broadcast_to(self.mask_tokens, (batch, total_patches, d_dec))
            token_idx = jax.vmap(
                lambda m: jnp.nonzero(m, size=k_patches, fill_value=0)[0]
            )(unmasked)  # (B, k_patches)
            x_full = jax.vmap(lambda xf, idx, xb: xf.at[idx].set(xb))(x_full, token_idx, x)
            return x_full

        def _scatter_into_full():
            unmasked = (mask == 0)  # Unmasked positions are where mask == 0
            x_full = jnp.broadcast_to(self.mask_tokens, (batch, total_patches, d_dec))
            token_idx = jax.vmap(  # Indices of unmasked patches per batch (size fixed to k_patches)
                lambda m: jnp.nonzero(m, size=k_patches, fill_value=0)[0]
            )(unmasked)  # (B, k_patches)
            # Scatter: x_full[b, token_idx[b, :], :] = x[b, :, :]
            x_full = jax.vmap(lambda xf, idx, xb: xf.at[idx].set(xb))(x_full, token_idx, x)
            return x_full

        x = jax.lax.cond(k_patches == total_patches, _already_full, _scatter_into_full)

        # Add positional embeddings
        x = x + self.pos_embeds.astype(x.dtype)

        # Transformer decode
        for block in self.vit_blocks:
            x = block(x, deterministic=deterministic)
        x = self.norm(x)
        return x
