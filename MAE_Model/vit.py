import jax
import jax.numpy as jnp
from flax import linen as nn

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) for Flax/JAX.

    Drops entire residual branches per-example. This is commonly used in ViTs.
    """
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        if self.drop_prob <= 0.0 or deterministic:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Per-example mask, broadcast over remaining dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng("dropout")
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
        return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))

class MLP(nn.Module):
    """Transformer MLP / FFN block."""
    embed_dim: int
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        hidden_dim = int(self.embed_dim * self.mlp_ratio)

        x = nn.Dense(hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)

        x = nn.Dense(self.embed_dim)(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)
        return x

class VitBlock(nn.Module):
    """Standard pre-norm Vision Transformer block (Flax/Linen).

    Expected input shape: (batch, tokens, embed_dim)
    """
    embed_dim: int
    heads: int
    mlp_ratio: float = 4.0
    attn_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    path_drop_rate: float = 0.0
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """
        ViT block includes multi headed self attention and multi-layer perception (mlp) layer,
        each with prenorm and residual/skip connection.

        Args:
            x (Tensor): Shape (batch, num_patches, embed_dim)

        Returns:
            x (Tensor): Same shape as input (batch, num_patches, embed_dim)
        """
        # Attention block
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.heads,
            dropout_rate=self.attn_drop_rate,
        )(x, x, deterministic=deterministic)
        x = DropPath(drop_prob=self.path_drop_rate)(x, deterministic=deterministic)
        x = residual + x

        # MLP block
        residual = x
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = MLP(
            embed_dim=self.embed_dim,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.mlp_drop_rate,
        )(x, deterministic=deterministic)
        x = DropPath(drop_prob=self.path_drop_rate)(x, deterministic=deterministic)
        x = residual + x

        return x