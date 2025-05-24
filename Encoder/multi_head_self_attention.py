import torch
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import transforms
from torch import Tensor
import torch.nn.functional as F
import os, cv2, math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim = 128, num_heads = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = int(self.embedding_dim / self.num_heads)
        
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim)

    def forward(self, x: Tensor):
        # Get Q, K, V and distribute among the heads
        batch, num_patches, _ = x.size()
        qkv = self.qkv_projection(x) # Q, K, V for all heads
        qkv = qkv.view(batch, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch, num_heads, num_patches, embedding_dim/num_heads)
        Q, K, V = qkv[0], qkv[1], qkv[2] # Q = [q_head1, q_head2, ...], and q_head1 = [section of q_patch1, section of q_patch2, ...]
        
        # Calculate attention for each head, assume Q, K, V have shape (batch, num_heads, num_patches, head_dim)
        all_attention = [] # Collects attention from different heads
        for head in range(self.num_heads):
            Qh, Kh, Vh = Q[:, head], K[:, head], V[:, head]
            attention_weights = F.softmax(torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.head_dim))
            attention_score = torch.matmul(attention_weights, Vh)
            all_attention.append(attention_score)
        
        final_attention = torch.cat(all_attention, dim=-1) # Concat along the final dimension
        return final_attention # shape = (batch, num_patches, embedding_dim)
            
                    
        
        
        