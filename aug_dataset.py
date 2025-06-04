import os
import numpy as np
import random
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class Augment:
    def __init__(self):
        self.directory = 'Dataset/Train'

    def apply_augmentation_pair(self, left_img, right_img, mode):
        if mode == "noise":
            noise = np.random.normal(0, 10, left_img.shape).astype(np.int16)
            left_img = np.clip(left_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            right_img = np.clip(right_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif mode == "brightness":
            shift = random.randint(-40, 40)
            left_img = np.clip(left_img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            right_img = np.clip(right_img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        elif mode == "color":
            factor = np.random.uniform(0.8, 1.2, 3)
            left_img = np.clip(left_img * factor, 0, 255).astype(np.uint8)
            right_img = np.clip(right_img * factor, 0, 255).astype(np.uint8)
        return left_img, right_img

    def augment_pair(self, file):
        left_dir = os.path.join(self.directory, 'LeftCam')
        right_dir = os.path.join(self.directory, 'RightCam')
        idx = file.split('.')[0].split('_')[-1]
        left_path = os.path.join(left_dir, file)
        right_path = os.path.join(right_dir, file.replace("left", "right"))

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        outputs = []
        for i, aug in enumerate(["noise", "brightness", "color"], start=1):
            augL, augR = self.apply_augmentation_pair(left_img.copy(), right_img.copy(), aug)
            out_left_name = os.path.join(left_dir, f"left_{idx}_aug{i}.png")
            out_right_name = os.path.join(right_dir, f"right_{idx}_aug{i}.png")
            outputs.append((out_left_name, augL, out_right_name, augR))
        return outputs

    def augment_dataset(self):
        left_dir = os.path.join(self.directory, 'LeftCam')
        left_files = sorted([f for f in os.listdir(left_dir) if f.endswith('.png')])

        print("Augmenting stereo dataset with noise, brightness, and color shifts...")
        with ProcessPoolExecutor() as executor:
            all_outputs = list(tqdm(executor.map(self.augment_pair, left_files), total=len(left_files)))

        for batch in all_outputs:
            for out_left_name, augL, out_right_name, augR in batch:
                cv2.imwrite(out_left_name, augL)
                cv2.imwrite(out_right_name, augR)

        print("Augmentation complete")
