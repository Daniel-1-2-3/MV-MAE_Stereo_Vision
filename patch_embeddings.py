from PIL import Image
import torch
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import transforms
from torch import Tensor
import os

class PatchEmbedding(nn.Module):
    def __init__ (self, batch_size, in_channels, patch_size, embedding_dim, img_size):
        super().__init__()
        num_patches = int((img_size / patch_size) ** 2)
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embedding_dim),
        )
        self.pos_embeddings = nn.Parameter(
            torch.randn(batch_size, num_patches, embedding_dim)
        )

    def forward(self, x: Tensor):
        x = self.projection(x)
        x += self.pos_embeddings
        return x # Returns an array containing embedding for each patch

if __name__ == "__main__":
    patch_embed = PatchEmbedding()    

    img = Image.open(os.path.join(os.getcwd(), 'TestImg1.jpg')).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    sample_tensor = transform(img).unsqueeze(0)
    print(sample_tensor.size())
    
    print(patch_embed.forward(sample_tensor).size())
    