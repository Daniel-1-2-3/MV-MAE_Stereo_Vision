import os
import numpy as np
import random
import cv2
import tqdm

class Augment:
    def __init__(self):
        self.directory='Dataset/Train'

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

    def augment_dataset(self):
        left_dir = os.path.join(self.directory, 'LeftCam')
        right_dir = os.path.join(self.directory, 'RightCam')

        left_files = sorted([f for f in os.listdir(left_dir) if f.endswith('.png')])
        
        print("Augmenting stereo dataset with noise, brightness, and color shifts...")
        for f in tqdm(left_files):
            idx = f.split('.')[0].split('_')[-1]
            left_path = os.path.join(left_dir, f)
            right_path = os.path.join(right_dir, f.replace("left", "right"))

            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)

            for i, aug in enumerate(["noise", "brightness", "color"], start=1):
                augL, augR = self.apply_augmentation_pair(left_img.copy(), right_img.copy(), aug)

                out_left_name = f"left_{idx}_aug{i}.png"
                out_right_name = f"right_{idx}_aug{i}.png"

                cv2.imwrite(os.path.join(left_dir, out_left_name), augL)
                cv2.imwrite(os.path.join(right_dir, out_right_name), augR)

        print("Augmentation complete")

