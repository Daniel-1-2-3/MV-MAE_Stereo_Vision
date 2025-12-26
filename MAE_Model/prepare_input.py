import jax.numpy as jp

class Prepare:
    @staticmethod
    def fuse_normalize(imgs):
        """
        Args:
            imgs: [left, right], each strictly (H, W, 3), dtype uint8 or float32.
                  If float32 in [0,1].

        Returns:
            x: (1, H, 2W, 3) float32 normalized
        """
        left, right = imgs

        # Fuse along width: (H, 2W, 3)
        x = jp.concatenate([left, right], axis=1).astype(jp.float32)

        mean = jp.array([0.51905, 0.47986, 0.48809], dtype=jp.float32).reshape(1, 1, 3)
        std  = jp.array([0.17454, 0.20183, 0.19598], dtype=jp.float32).reshape(1, 1, 3)
        x = (x - mean) / std

        # Add batch dim: (1, H, 2W, 3)
        return x[None, ...]
