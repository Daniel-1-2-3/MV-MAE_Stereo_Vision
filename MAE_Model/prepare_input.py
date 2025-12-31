import torch

class Prepare:
    _cached = {}  # (device, dtype) -> (mean, std)

    @staticmethod
    def _stats(device, dtype):
        key = (device, dtype)
        if key not in Prepare._cached:
            mean = torch.tensor([0.51905, 0.47986, 0.48809], device=device, dtype=dtype).view(1, 1, 1, 3)
            std  = torch.tensor([0.17454, 0.20183, 0.19598], device=device, dtype=dtype).view(1, 1, 1, 3)
            Prepare._cached[key] = (mean, std)
        return Prepare._cached[key]

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        was_u8 = (x.dtype == torch.uint8)
        x = x.float()
        if was_u8:
            x = x / 255.0
        mean, std = Prepare._stats(x.device, x.dtype)
        return (x - mean) / std