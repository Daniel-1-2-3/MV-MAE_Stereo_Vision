from patch_embeddings import PatchEmbedding
from TransformerLayer.multi_head_self_attention import MultiHeadSelfAttention
from TransformerLayer.feed_fwd import FeedForward
from decoder_input_prepare import DecoderInputPreparation

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import cv2
import argparse
class Model(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_channels=3,
                 encoder_embed_dim=768, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_num_heads=8):
        super().__init__()
        self.img_size, self.in_channels = img_size, in_channels
        self.patch_size = patch_size
        self.num_patches = int(img_size / patch_size)**2
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, encoder_embed_dim, img_size)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttention(encoder_embed_dim, encoder_num_heads),
                FeedForward(encoder_embed_dim)
            ) for _ in range(12) # 12 layers of the encoder block
        ])
        self.prepare_decoder_in = DecoderInputPreparation(img_size, patch_size, encoder_embed_dim, decoder_embed_dim)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttention(decoder_embed_dim, decoder_num_heads),
                FeedForward(decoder_embed_dim)
            ) for _ in range(4) # Lighter, only 4 layers of decoder block
        ])
        self.reconstruct_projection = nn.Linear(self.decoder_embed_dim, self.in_channels * self.patch_size**2)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.patch_embed.forward(x)
            for encoder_block in self.encoder:
                x = encoder_block(x)
        
            x = self.prepare_decoder_in.forward(x)
            for decoder_block in self.decoder:
                x = decoder_block(x)
            return x # shape = (batch_size, num_patches(both images), vector dimension of each patch embed)
    
    def get_mse_loss(self, x, img_masked, show = False):
        grid_size = self.img_size // self.patch_size # Num of patches along one dimension
        
        current_batch_size = x.shape[0]
        reconstructed = self.reconstruct_projection(x[:, self.num_patches:, :])
        reconstructed = reconstructed.view(current_batch_size, grid_size, grid_size, self.patch_size, self.patch_size, self.in_channels)
        reconstructed = reconstructed.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(current_batch_size, self.in_channels, self.img_size, self.img_size)
        
        if show:
            # Convert from (C, H, W) to (H, W, C), scale to 0–255, and convert to uint8
            rec_img = reconstructed[0].permute(1, 2, 0).cpu().numpy()
            gt_img = img_masked[0].permute(1, 2, 0).cpu().numpy()

            rec_img = (rec_img * 255).clip(0, 255).astype(np.uint8)
            gt_img = (gt_img * 255).clip(0, 255).astype(np.uint8)

            combined = np.hstack((gt_img, rec_img))
            cv2.imshow("Ground Truth (Left) | Reconstructed (Right)", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Calculated MSE loss per pixel
        mse_loss = F.mse_loss(reconstructed, img_masked)
        return mse_loss
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()
    
    left_image_path = os.path.join(os.getcwd(), 'Dataset', 'LeftCam', 'left_view_0.png')
    left_img = Image.open(left_image_path).convert("RGB")
    right_image_path = os.path.join(os.getcwd(), 'Dataset', 'RightCam', 'right_view_0.png')
    right_img = Image.open(right_image_path).convert("RGB")

    img_size = 256
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    visible_tensor = transform(left_img).unsqueeze(0)  
    masked_tensor = transform(right_img).unsqueeze(0) 

    model = Model()
    model.eval()
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(os.path.join('Weights', args.weights), map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    
    with torch.no_grad():
        decoder_output = model(visible_tensor)
        mse_loss = model.get_mse_loss(decoder_output, masked_tensor, show=True)
        print("MSE Loss:", mse_loss.item())

        

    