from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import functools

from MAE_Model.encoder import ViTMaskedEncoder
from MAE_Model.decoder import ViTMaskedDecoder

class MAEModel(nn.Module):
    nviews: int = 2
    patch_size: int = 8
    encoder_embed_dim: int = 768
    decoder_embed_dim: int = 512
    encoder_heads: int = 16
    decoder_heads: int = 16
    in_channels: int = 3
    img_h_size: int = 84
    img_w_size: int = 84
    masking_ratio: float = 0.75

    # Keep these matching your encoder/decoder defaults if you rely on them elsewhere.
    encoder_depth: int = 8
    decoder_depth: int = 4

    def setup(self):
        self.img_w_fused = self.nviews * self.img_w_size
        self.num_patches = int(
            (self.img_h_size * self.img_w_size) // (self.patch_size ** 2) * self.nviews
        )

        self.encoder = ViTMaskedEncoder(
            nviews=self.nviews,
            patch_size=self.patch_size,
            embed_dim=self.encoder_embed_dim,
            in_channels=self.in_channels,
            img_h_size=self.img_h_size,
            img_w_fused_size=self.img_w_fused,
            heads=self.encoder_heads,
            masking_ratio=self.masking_ratio,
            depth=self.encoder_depth,
        )
        self.decoder = ViTMaskedDecoder(
            nviews=self.nviews,
            patch_size=self.patch_size,
            encoder_embed_dim=self.encoder_embed_dim,
            img_h_size=self.img_h_size,
            img_w_size=self.img_w_size,
            decoder_embed_dim=self.decoder_embed_dim,
            in_channels=self.in_channels,
            heads=self.decoder_heads,
            depth=self.decoder_depth,
        )
        self.out_proj = nn.Dense(features=self.patch_size ** 2 * self.in_channels)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool):
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
                                    actor in the pipeline. It is the input, masked, passed through the encoder
        """
        z, mask = self.encoder(x, True, deterministic=deterministic)  # (B, kept, D_enc), (B, total_patches)
        decoded = self.decoder(z, mask, deterministic=deterministic)  # (B, total_patches, D_dec)
        out = self.out_proj(decoded)  # (B, total_patches, patch_size^2 * C)

        return out, mask, z

    def encoder_with_masking(self, x, deterministic: bool):
        z, mask = self.encoder(x, True, deterministic=deterministic)
        return z, mask

    def encoder_no_masking(self, x, deterministic: bool):
        z, _ = self.encoder(x, False, deterministic=deterministic)
        return z

    def compute_loss(self, out: jnp.ndarray, truth: jnp.ndarray, mask: jnp.ndarray):
        # Average only over masked patches (mask==1)
        mask = mask.astype(jnp.float32)
        truth_patchified = self.patchify(truth)

        loss_per_patch = jnp.mean((out - truth_patchified) ** 2, axis=-1)  # (B, total_patches)
        den = jnp.sum(mask)

        # Avoid Python branching on traced values
        loss = jnp.sum(loss_per_patch * mask) / jnp.where(den > 0.0, den, loss_per_patch.size)
        return loss

    def patchify(self, x: jnp.ndarray):
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

        p = self.patch_size
        w_view = w_total // self.nviews
        h = height // p
        w = w_view // p

        # Split along width into views into (b, h, w, c) x nviews
        # (B, H, W_total, C) -> (B, H, V, W_view, C) -> (B, V, H, W_view, C)
        x = x.reshape(batch, height, self.nviews, w_view, in_channels)
        x = jnp.transpose(x, (0, 2, 1, 3, 4))

        # Patchify each view
        # (B, V, H, W_view, C) -> (B, V, h, p, w, p, C)
        x = x.reshape(batch, self.nviews, h, p, w, p, in_channels)
        x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6)) # -> (B, V, h, w, p, p, C)
        x = x.reshape(batch, self.nviews, h * w, p * p * in_channels) # -> (B, V, h*w, p*p*C)

        # Concatenate views along patch dimension: (B, total_patches, patch_dim)
        x = x.reshape(batch, self.nviews * h * w, p * p * in_channels)
        return x

    def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, total_patches, patch_size**2 * C)
        Returns:
            imgs: (B, C, H, W_fused) with views stitched horizontally
        """
        batch, total_patches, dim = x.shape
        c = dim // (self.patch_size ** 2)

        p = self.patch_size
        patches_per_view = total_patches // self.nviews
        h = self.img_h_size // p
        w = self.img_w_size // p

        # (B, total_patches, dim) -> (B, V, patches_per_view, dim)
        x = x.reshape(batch, self.nviews, patches_per_view, dim)
        # (B, V, h*w, p*p*C) -> (B, V, h, w, p, p, C)
        x = x.reshape(batch, self.nviews, h, w, p, p, c)
        x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6)) # -> (B, V, h, p, w, p, C)
        x = x.reshape(batch, self.nviews, h * p, w * p, c) # -> (B, V, H, W, C)

        # Stitch views along width: (B, V, H, W, C) -> (B, H, V, W, C) -> (B, H, W_fused, C)
        x = jnp.transpose(x, (0, 2, 1, 3, 4)).reshape(batch, h * p, self.nviews * w * p, c)
        return jnp.transpose(x, (0, 3, 1, 2)) # (B, C, H, W_fused)

def make_encoder_apply_fns(model: MAEModel):
    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def encoder_masked_apply(variables, x, *, deterministic: bool, rngs):
        return model.apply(
            variables,
            x,
            deterministic=deterministic,
            method=MAEModel.encoder_with_masking,
            rngs=rngs,
        )

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def encoder_unmasked_apply(variables, x, *, deterministic: bool, rngs):
        return model.apply(
            variables,
            x,
            deterministic=deterministic,
            method=MAEModel.encoder_no_masking,
            rngs=rngs,
        )

    return encoder_masked_apply, encoder_unmasked_apply