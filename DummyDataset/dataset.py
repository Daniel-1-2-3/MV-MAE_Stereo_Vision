import os
from torch.utils.data import Dataset
from PIL import Image

class StereoImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Pytorch-compatible dataset for use with DataLoader

        Args:
            root_dir (str): Directory of the dataset root folder, which contains the train and val sets
            transform (torchvision.transforms.Compose): Transformations applied to all images in the dataset, such as resizing
                                                        and converting RGB to tensor. Defaults to None.
        """
        self.left_dir = os.path.join(root_dir, "LeftCam")
        self.right_dir = os.path.join(root_dir, "RightCam")
        self.left_images = sorted([f for f in os.listdir(self.left_dir) if f.endswith('.png')])
        self.right_images = sorted([f for f in os.listdir(self.right_dir) if f.endswith('.png')])
        assert len(self.left_images) == len(self.right_images), "Mismatch in left/right image counts"
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_path = os.path.join(self.left_dir, self.left_images[idx])
        right_path = os.path.join(self.right_dir, self.right_images[idx])

        left_img = Image.open(left_path).convert('RGB')
        right_img = Image.open(right_path).convert('RGB')

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img
