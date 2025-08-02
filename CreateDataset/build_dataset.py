import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import os
import hashlib
from tqdm import tqdm
from itertools import product
from pathlib import Path
from multiprocessing import Pool, cpu_count
from aug_dataset import Augment

class StereoCamera:
    def __init__(self, r, cube_pos):
        self.cube_pos = cube_pos
        self.r = r
        self.yaw, self.pitch = 0.0, 20.0
        self.update_xyz()
        self.width, self.height = 256, 256
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=self.width/self.height, nearVal=0.1, farVal=100)
        self.img_left, self.img_right = None, None

    def render(self):
        left_target, right_target = self.get_cam_target_poses()
        view_left = p.computeViewMatrix([self.x_left, self.y_left, self.z_left], list(left_target), [0, 0, 1])
        view_right = p.computeViewMatrix([self.x_right, self.y_right, self.z_right], list(right_target), [0, 0, 1])

        _, _, rgb_left, _, _ = p.getCameraImage(self.width, self.height, view_left, self.projection_matrix, renderer=p.ER_TINY_RENDERER)
        _, _, rgb_right, _, _ = p.getCameraImage(self.width, self.height, view_right, self.projection_matrix, renderer=p.ER_TINY_RENDERER)

        self.img_left = np.reshape(rgb_left, (self.height, self.width, 4))[:, :, :3].astype(np.uint8)
        self.img_right = np.reshape(rgb_right, (self.height, self.width, 4))[:, :, :3].astype(np.uint8)

    def get_cam_target_poses(self):
        left_cam_point = [self.x_left, self.y_left, self.z_left]
        right_cam_point = [self.x_right, self.y_right, self.z_right]
        cam_midpoint = (np.array(left_cam_point) + np.array(right_cam_point)) / 2
        look_dir = (np.array(self.cube_pos) - cam_midpoint)
        look_dir /= np.linalg.norm(look_dir)
        return left_cam_point + look_dir, right_cam_point + look_dir

    def update_xyz(self):
        z = math.sin(math.radians(self.pitch)) * self.r
        r2 = math.cos(math.radians(self.pitch)) * self.r
        x = math.cos(math.radians(self.yaw)) * r2
        y = math.sin(math.radians(self.yaw)) * r2
        midpoint = np.array([x, y, z])
        look_dir = (np.array(self.cube_pos) - midpoint)
        look_dir /= np.linalg.norm(look_dir)
        right_vec = np.cross(look_dir, [0, 0, 1])
        right_vec /= np.linalg.norm(right_vec)
        d = 0.5
        left_eye = midpoint - (d / 2) * right_vec
        right_eye = midpoint + (d / 2) * right_vec
        self.x_left, self.y_left, self.z_left = left_eye
        self.x_right, self.y_right, self.z_right = right_eye

def render_combination(args):
    z, pitch, yaw = args
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    ground_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.1])
    ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1])
    p.createMultiBody(baseCollisionShapeIndex=ground_shape, baseVisualShapeIndex=ground_visual, basePosition=[0, 0, -0.1])

    cube_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[0.2, 0.2, 0.7, 1.0])
    p.createMultiBody(
        baseVisualShapeIndex=cube_visual,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
        basePosition=[0, 0, 0.4])

    sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.7, 0.2, 0.2, 1.0])
    p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.35),
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0.75, 0, 0.35])

    cam = StereoCamera(3, [0, 0, 0])
    cam.yaw, cam.pitch, cam.r = yaw, pitch, z / 2
    cam.update_xyz()
    p.stepSimulation()
    cam.render()
    result = (cam.img_left.copy(), cam.img_right.copy())
    p.disconnect()
    return result

def save_and_split_dataset(all_images, base_dir='Dataset', split_ratio=0.7):
    print("Removing duplicates and splitting dataset...")
    seen_hashes = set()
    unique_images = []

    for left, right in tqdm(all_images, desc="Removing duplicates..."):
        combined = np.concatenate((left, right), axis=1)
        hash_val = hashlib.md5(combined.tobytes()).hexdigest()
        if hash_val not in seen_hashes:
            seen_hashes.add(hash_val)
            unique_images.append((left, right))

    np.random.shuffle(unique_images)
    split_index = int(len(unique_images) * split_ratio)
    train_set, val_set = unique_images[:split_index], unique_images[split_index:]

    def save_images(pairs, left_dir, right_dir):
        Path(left_dir).mkdir(parents=True, exist_ok=True)
        Path(right_dir).mkdir(parents=True, exist_ok=True)
        for i, (l, r) in enumerate(pairs):
            cv2.imwrite(os.path.join(left_dir, f"left_{i}.png"), l)
            cv2.imwrite(os.path.join(right_dir, f"right_{i}.png"), r)

    save_images(train_set, f"{base_dir}/Train/LeftCam", f"{base_dir}/Train/RightCam")
    save_images(val_set, f"{base_dir}/Val/LeftCam", f"{base_dir}/Val/RightCam")
    print(f"Done. Saved {len(train_set)} training pairs and {len(val_set)} validation pairs.")

if __name__ == "__main__":
    zoom_range = range(4, 11)
    pitch_range = range(2, 90, 2)
    yaw_range = range(0, 360, 10)
    combinations = list(product(zoom_range, pitch_range, yaw_range))

    with Pool(cpu_count()) as pool:
        images = list(tqdm(pool.imap(render_combination, combinations), total=len(combinations), desc="Collecting images"))

    save_and_split_dataset(images)
    aug = Augment()
    aug.augment_dataset()
    
    
