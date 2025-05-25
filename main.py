from patch_embeddings import PatchEmbedding
from TransformerLayer.multi_head_self_attention import MultiHeadSelfAttention
from TransformerLayer.feed_fwd import FeedForward
from decoder_input_prepare import DecoderInputPreparation

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import os, cv2
import numpy as np
import torch.nn.functional as F

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

class Model(nn.Module):
    def __init__(self, img_size=256, patch_size=8, batch_size=1, in_channels=3,
                 encoder_embed_dim=768, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_num_heads=8):
        super().__init__()
        self.img_size, self.patch_size, self.in_channels = img_size, patch_size, in_channels
        self.num_patches = int(img_size / patch_size)**2
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        self.patch_embed = PatchEmbedding(batch_size, in_channels, patch_size, 
                                          encoder_embed_dim, img_size)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttention(encoder_embed_dim, encoder_num_heads),
                FeedForward(encoder_embed_dim)
            ) for _ in range(12) # 12 layers of the encoder block
        ])
        self.prepare_decoder_in = DecoderInputPreparation(batch_size, img_size, patch_size,
                                                    encoder_embed_dim, decoder_embed_dim)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttention(decoder_embed_dim, decoder_num_heads),
                FeedForward(decoder_embed_dim)
            ) for _ in range(4) # Lighter, only 4 layers of decoder block
        ])
    
    def forward(self, x):
        with torch.no_grad():
            x = self.patch_embed.forward(x)
            for encoder_block in self.encoder:
                x = encoder_block(x)
        
            x = self.prepare_decoder_in.forward(x)
            for decoder_block in self.decoder:
                x = decoder_block(x)
            return x # shape = (batch_size, num_patches(both images), vector dimension of each patch embed)
    
    def get_reconstruct_loss(self, x, img_masked):
        projection = nn.Linear(self.decoder_embed_dim, self.in_channels * self.patch_size**2)
        grid_size = self.img_size // self.patch_size # Num of patches along one dimension
        
        reconstructed = projection(x[:, self.num_patches:, :])
        reconstructed = reconstructed.view(1, grid_size, grid_size, self.patch_size, self.patch_size, self.in_channels)
        reconstructed = reconstructed.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(1, 3, self.img_size, self.img_size)
        
        # Calculated MSE loss per pixel
        mse_loss = F.mse_loss(reconstructed, img_masked)
        print('MSE loss', mse_loss)
        
        # Img for display
        img = reconstructed.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = (img * 255).astype(np.uint8)

        cv2.imshow("Reconstructed View", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return mse_loss

if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    img_visible = Image.open(os.path.join(os.getcwd(), 'TestImg1.jpg')).convert('RGB')
    img_visible = transform(img_visible).unsqueeze(0)
    img_masked = Image.open(os.path.join(os.getcwd(), 'TestImg2.jpg')).convert('RGB')
    img_masked = transform(img_masked).unsqueeze(0)
  
    model = Model()
    x = model(img_visible) # shape = (batch_size, num_patches * 2, decoder_dim)
    
    print(x.size())
    model.get_reconstruct_loss(x, img_masked)