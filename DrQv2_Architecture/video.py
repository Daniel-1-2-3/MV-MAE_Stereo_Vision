# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)
    def save(self, path):
        """
        Disabled video saving on cluster to avoid ffmpeg issues.
        Training will continue; no videos will be written.
        """
        # Clear frames so we don't keep them in memory
        self.frames = []
        return
    """
    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name

            frames_uint8 = []
            for frame in self.frames:
                # If the env returned normalized float32 frames (your Sawyer env),
                # de-normalize using the same stats as Prepare.fuse_normalize
                if frame.dtype != np.uint8:
                    # frame shape: (H, 2W, 3), float32, ~[-4, 4]
                    mean = np.array([0.51905, 0.47986, 0.48809], dtype=np.float32).reshape(1, 1, 3)
                    std  = np.array([0.17454, 0.20183, 0.19598], dtype=np.float32).reshape(1, 1, 3)

                    img = frame * std + mean        # back to roughly [0, 1]
                    img = np.clip(img, 0.0, 1.0)
                    img = (img * 255.0).astype(np.uint8)
                else:
                    # Already uint8 [0, 255] (other envs)
                    img = frame

                frames_uint8.append(img)

            imageio.mimsave(str(path), frames_uint8, fps=self.fps, codec='mpeg4')
    """
class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
