import torch

class Prepare:
    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize fused image tensor.

        Args:
            x: (B, H, 2W, 3) uint8 or float tensor.
        Returns:
            (B, H, 2W, 3) float32 normalized.
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        mean = torch.tensor([0.51905, 0.47986, 0.48809], device=x.device).view(1, 1, 1, 3)
        std  = torch.tensor([0.17454, 0.20183, 0.19598], device=x.device).view(1, 1, 1, 3)

        return (x - mean) / std