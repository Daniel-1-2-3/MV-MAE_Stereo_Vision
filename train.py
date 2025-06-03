import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import os
import argparse

from model import Model
from dataset import StereoImageDataset

class Train():
    def __init__(self, img_size, patch_size, batch_size, in_channels,
                 encoder_embed_dim, encoder_num_heads,
                 decoder_embed_dim, decoder_num_heads):

        self.model = Model(img_size, patch_size, in_channels, encoder_embed_dim,
                           encoder_num_heads, decoder_embed_dim, decoder_num_heads)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), 
        ])
        print("Loading dataset...")
        dataset = StereoImageDataset(root_dir='Dataset', transform=self.transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def train(self, num_epochs, lr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f'Running on {device}')
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for left, right in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}", leave=False):
                left = left.to(device)
                right = right.to(device)
                
                visible, masked = random.choice([(left, right), (right, left)])
                optimizer.zero_grad()
                x = self.model(visible)
                mse_loss = self.model.get_mse_loss(x, masked.to(device))
                mse_loss.backward()
                optimizer.step()
                
                total_loss += mse_loss.item()
                
            avg_loss = total_loss / len(self.dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {round(avg_loss, 4)}')

            if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
                self.save_weights(epoch)
    
    def save_weights(self, epoch):
        os.makedirs('Weights', exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join('Weights', f'weights_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--encoder_embed_dim", type=int, default=768)
    parser.add_argument("--encoder_num_heads", type=int, default=12)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0004) 
    args = parser.parse_args()

    trainer = Train(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_num_heads=args.decoder_num_heads,
    )
    
    trainer.train(num_epochs=args.num_epochs, lr=args.lr)
