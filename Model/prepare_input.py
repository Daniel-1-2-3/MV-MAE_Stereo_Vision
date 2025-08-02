import torch

class Prepare():
    
    @staticmethod
    def fuse_normalize(imgs: list[torch.Tensor]):
        """
        Fuse images along their height and normalize

        Args:
            imgs (list[Tensor]):    Two images (stereo vision), both with 
                                    shape (batch, channels, height, width)
        Returns:
            x (Tensor): A single tensor of shape (batch, height, width_total, channels)
        """
        # Transpose to (b, h, w, c)
        left = imgs[0].permute(0, 2, 3, 1)
        right = imgs[1].permute(0, 2, 3, 1)

        x = torch.cat([left, right], dim=2) # Fuse (b, h, w_total, c)
        
        # Normalize
        mean = torch.tensor([0.51905, 0.47986, 0.48809], device=x.device).view(1, 1, 1, 3)
        std = torch.tensor([0.17454, 0.20183, 0.19598], device=x.device).view(1, 1, 1, 3)
        x = (x - mean) / std
        return x.to(imgs[0].device)