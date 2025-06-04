import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import os
import argparse
import requests

from model import Model
from dataset import StereoImageDataset

class Train():
    def __init__(self, img_size, patch_size, batch_size, in_channels,
                 encoder_embed_dim, encoder_num_heads,
                 decoder_embed_dim, decoder_num_heads, server):

        self.server = server
        self.model = Model(img_size, patch_size, in_channels, encoder_embed_dim,
                           encoder_num_heads, decoder_embed_dim, decoder_num_heads)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), 
        ])
        print("Loading dataset...")

        train_dir = 'Dataset/Train'
        val_dir = 'Dataset/Val'

        train_dataset = StereoImageDataset(root_dir=train_dir, transform=self.transform)
        val_dataset = StereoImageDataset(root_dir=val_dir, transform=self.transform)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def train(self, num_epochs, lr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f'Running on {device}')
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for left, right in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False):
                left, right = left.to(device), right.to(device)
                visible, masked = random.choice([(left, right), (right, left)])
                
                optimizer.zero_grad()
                x = self.model(visible)
                mse_loss = self.model.get_mse_loss(x, masked)
                mse_loss.backward()
                optimizer.step()
                
                total_loss += mse_loss.item()
            
            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.evaluate(device)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs or epoch == 0:
                print('Saving checkpoint...')
                self.save_weights(epoch)
                self.update_and_save_losses(epoch, avg_train_loss, avg_val_loss)
    
    def evaluate(self, device):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for left, right in self.val_loader:
                left, right = left.to(device), right.to(device)
                visible, masked = random.choice([(left, right), (right, left)])
                x = self.model(visible)
                loss = self.model.get_mse_loss(x, masked)
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.val_loader)

    def save_weights(self, epoch):
        os.makedirs('Results', exist_ok=True)
        filename = f'weights_epoch_{epoch + 1}.pth'
        filepath = os.path.join('Results', filename)
        torch.save(self.model.state_dict(), filepath)
        
        # Only for hosted GPU on RunPod
        try:
            with open(filepath, 'rb') as f:
                response = requests.post(
                    f"{self.server}/upload", 
                    files={"file": (filename, f, "application/octet-stream")}
                )
            print(f"Server response: {response.json()}")
        except Exception:
            print("Server may be offline")
    
    def update_and_save_losses(self, epoch, train_loss, val_loss):
        os.makedirs("Results", exist_ok=True)
        log_path = os.path.join("Results", "losses.txt")
        
        line = f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n"
        with open(log_path, "a") as f:
            f.write(line)
        
        # Only for hosted GPU on RunPod
        try:
            with open(log_path, "rb") as f:
                response = requests.post(
                    f"{self.server}/upload",
                    files={"file": ("losses.txt", f, "text/plain")}
                )
            print(f"Loss recorded. Server response: {response.json()}")
        except Exception:
            print("Server may be offline")
        
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
    parser.add_argument("--server", type=str)
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
        server=args.server
    )
    
    trainer.train(num_epochs=args.num_epochs, lr=args.lr)
