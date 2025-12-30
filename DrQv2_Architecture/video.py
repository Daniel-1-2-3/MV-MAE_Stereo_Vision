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
        self.enabled = False

        # Normalization stats (from Prepare.normalize)
        self.mean = np.array([0.51905, 0.47986, 0.48809], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.17454, 0.20183, 0.19598], dtype=np.float32).reshape(1, 1, 3)

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = (self.save_dir is not None and enabled)
        if self.enabled:
            self.record(env)

    def record(self, env):
        if not self.enabled:
            return
        
        frame = env._get_img_obs()
        # frame: float32, (H, 2W, 3) normalized approx [-4, 4]

        if frame is not None:
            if frame.dtype != np.uint8:
                img = frame * self.std + self.mean  # back to [0,1]
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
            else:
                img = frame

            img = cv2.resize(img, (self.render_size*2, self.render_size))
            self.frames.append(img)

    def save(self, file_name):
        if not self.enabled:
            self.frames = []
            return
            
        path = self.save_dir / file_name
            
        # More reliable on HPC: save GIF (no ffmpeg needed)
        imageio.mimsave(str(path.with_suffix(".gif")), self.frames, fps=self.fps)

        self.frames = []