from patch_embeddings import PatchEmbedding
from Encoder.multi_head_self_attention import MultiHeadSelfAttention

from PIL import Image
from torch import nn
from torchvision import transforms
import os

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.mhsa = MultiHeadSelfAttention()
    
    def forward(self, x):
        x = self.patch_embed.forward(x)
        x = self.mhsa.forward(x)
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
    