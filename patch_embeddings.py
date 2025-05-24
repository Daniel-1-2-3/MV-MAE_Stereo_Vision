from PIL import Image
import torch
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import transforms
from torch import Tensor
import os, cv2, math

class PatchEmbedding(nn.Module):
    def __init__ (self, in_channels = 3, patch_size = 8, embedding_dim = 128, img_size = 512):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = int((img_size / patch_size) ** 2)
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels, self.embedding_dim),
        )
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches, self.embedding_dim)
        )

    def forward(self, x: Tensor):
        x = self.projection(x)
        x += self.pos_embeddings
        return x # Returns an array containing embedding for each patch

if __name__ == "__main__":
    patch_embed = PatchEmbedding()    

    img = Image.open(os.path.join(os.getcwd(), 'testImg.jpg')).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    sample_tensor = transform(img).unsqueeze(0)
    print(sample_tensor.size())
    
    print(patch_embed.forward(sample_tensor).size())
    