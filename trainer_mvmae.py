from DummyDataset.dataset import StereoImageDataset
from MAE_Model.prepare_input import Prepare
from MAE_Model.model import MAEModel
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

class MVMAETrainer():
    def __init__(self, 
            nviews=2,
            patch_size=8,
            encoder_embed_dim=768,
            decoder_embed_dim=512,
            encoder_heads=16,
            decoder_heads=16,
            in_channels=3,
            img_h_size=128,
            img_w_size=128, 
            lr=1e-4,
            epochs=50,
            batch_size=16,
            base_dataset_dir=""
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MAEModel(
            nviews=nviews,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            encoder_heads=encoder_heads,
            decoder_heads=decoder_heads,
            in_channels=in_channels,
            img_h_size=img_h_size,
            img_w_size=img_w_size,
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_w_size, img_h_size)),
            transforms.ToTensor(),
        ])
        
        print("Loading dataset...")
        train_dir, val_dir = os.path.join(base_dataset_dir, 'Train'), os.path.join(base_dataset_dir, 'Val')
        train_dataset = StereoImageDataset(root_dir=train_dir, transform=self.transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=6, pin_memory=True, drop_last=True)
        
        val_dataset = StereoImageDataset(root_dir=val_dir, transform=self.transform)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=6, pin_memory=True, drop_last=True)
        
        results_folder = os.path.join(os.getcwd(), 'results')
        if os.path.exists(results_folder): # Clear the folder
            shutil.rmtree(results_folder)
    
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003) 
        self.model.to(self.device)
        self.model.train()
        for epoch in range(0, self.epochs):
            total_loss = 0.0
            for x1, x2 in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                optimizer.zero_grad()
                x = Prepare.fuse_normalize([x1, x2])
                out, mask, encoder_nomask_x = self.model(x)
                loss = self.model.compute_loss(out, x, mask)
                loss.backward()
                optimizer.step()
                # self.debug(out, self.model)
                
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.evaluate()
            loss_log = f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            
            os.makedirs('results', exist_ok=True)
            # Save loss log
            print(loss_log)
            log_path = os.path.join(os.getcwd(), 'results', "losses.txt")
            with open(log_path, "a") as f:
                f.write(loss_log + "\n")
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(os.getcwd(), 'results', 'checkpoint.pt'))
            
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x1, x2 in self.val_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                x = Prepare.fuse_normalize([x1, x2])
                out, mask = self.model(x)
                loss = self.model.compute_loss(out, x, mask)
                total_loss += loss.item()
            self.model.render_reconstruction(out)
                
        self.model.train()
        return total_loss / len(self.val_loader)
        
    def debug(self, out, model, file_path='debug.txt'):
        with open(file_path, 'a') as f:
            f.write('Shape: {}\nd'.format(out.shape))
            f.write("Mean: {}\n".format(out.mean().item()))
            f.write("Std dev: {}\n".format(out.std().item()))
            f.write("NaNs: {}\n".format(torch.isnan(out).any().item()))
            
            cos = torch.nn.functional.cosine_similarity(out[:, 1:, :], out[:, 1:, :].mean(dim=1, keepdim=True), dim=-1)
            f.write("Mean cosine similarity to average patch: {}\n".format(cos.mean().item()))

            f.write("\n--- Weight Statistics ---\n")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    weight_min = param.data.min().item()
                    weight_max = param.data.max().item()
                    f.write(f"{name}: mean={weight_mean:.4e}, std={weight_std:.4e}, min={weight_min:.4e}, max={weight_max:.4e}\n")
            f.write("\n")
            
if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(), 'dataset')
    
    if not os.path.exists(dataset_folder): 
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('thisisdaniel12345/dataset', path=os.path.join(os.getcwd(), 'dataset'), unzip=True)
        
    trainer = MVMAETrainer(
        nviews=2,
        patch_size=8,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        encoder_heads=16,
        decoder_heads=16,
        in_channels=3,
        img_h_size=128,
        img_w_size=128, 
        lr=1e-4,
        epochs=50,
        batch_size=16,
        base_dataset_dir=os.path.join(os.getcwd(), 'dataset')
    )

    trainer.train()