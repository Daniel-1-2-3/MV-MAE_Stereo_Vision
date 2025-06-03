import torch
from torch import Tensor, nn

class DecoderInputPreparation(nn.Module):
    def __init__(self, img_size, patch_size, encoder_embed_dim, decoder_embed_dim):
        super().__init__()
        self.num_patches = int(img_size / patch_size) ** 2
        self.change_dim = nn.Linear(encoder_embed_dim, decoder_embed_dim) # The decoder is more lightweight than encoder
        
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, decoder_embed_dim)
        ) # Expanding on the single mask token at runtime saves memory
        
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches * 2, decoder_embed_dim) 
        ) # Decoder input after concatenation has this shape
    
    def forward(self, x: Tensor):
        current_batch_size = x.shape[0]
        x = self.change_dim(x)
        mask_tokens = self.mask_token.expand(current_batch_size, self.num_patches, -1)
        decoder_input = torch.cat([x, mask_tokens], dim=1)
        decoder_input += self.pos_embeddings
        return decoder_input
        
        
        