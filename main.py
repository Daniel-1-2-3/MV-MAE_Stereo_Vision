from patch_embeddings import PatchEmbedding
from Encoder.multi_head_self_attention import MultiHeadSelfAttention
from Encoder.feed_fwd import FeedForward

from PIL import Image
from torch import nn
from torchvision import transforms
import os

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.encoder = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttention(),
                FeedForward()
            ) for _ in range(2) # Standard ViT has 12, but temporarily shortened to 2 for speed
        ])
    
    def forward(self, x):
        x = self.patch_embed.forward(x)
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            print(f'Ran encoder block {i + 1}')
        return x
        
if __name__ == "__main__":
    model = Model()
    
    img = Image.open(os.path.join(os.getcwd(), 'TestImg.jpg')).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    x = transform(img).unsqueeze(0)
    print(x.size())
    
    x = model.forward(x)
    print(x.size())
    