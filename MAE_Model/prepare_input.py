import torch

class Prepare:
    _cached = {}  # device -> (mean_f32, std_f32)

    @staticmethod
    def _stats(device):
        if device not in Prepare._cached:
            mean = torch.tensor([0.51905, 0.47986, 0.48809], device=device, dtype=torch.float32).view(1,1,1,3)
            std  = torch.tensor([0.17454, 0.20183, 0.19598], device=device, dtype=torch.float32).view(1,1,1,3)
            Prepare._cached[device] = (mean, std)
        return Prepare._cached[device]

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and not x.is_contiguous():
            x = x.contiguous()
            
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) * (1.0 / 255.0)
        else:
            x = x.to(torch.float32)

        mean, std = Prepare._stats(x.device)
        return (x - mean) / st