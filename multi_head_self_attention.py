import torch
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import transforms
from torch import Tensor
import os, cv2, math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, patch_size = 8, embedding_dim = 128, img_size = 512, batch = 1, num_heads = 8):
        super().__init__()
        self.batch = batch
        self.patch_size = patch_size
        self.num_patches = int((img_size / patch_size) ** 2)
        self.embedding_dim = embedding_dim
        
        self.num_heads = num_heads
        self.head_dim = int(self.embedding_dim / self.num_heads)
        
        self.Q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.K = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.V = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def split_among_heads(self, x: Tensor):
        x = x.view(self.batch, self.num_patches, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3) # (batch, num_heads, num_patches, head_dim)
        return x
        