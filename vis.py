"""Image / video helpers shared by the scripts (no simulator imports).

Videos follow the pattern that works on display-less pods: frames are kept as
uint8 (H, W, 3) numpy arrays and written with imageio + ffmpeg (libx264); wandb
gets the finished mp4 file, so it needs no video encoder of its own. Frame sizes
are kept multiples of 16 so the encoder never has to resize.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def to_hwc(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """(3, H, W) or (H, W, 3) uint8 -> numpy (H, W, 3) uint8."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    return np.ascontiguousarray(img.astype(np.uint8))


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    return img.repeat(scale, axis=0).repeat(scale, axis=1)


def stereo_frame(stereo: torch.Tensor | np.ndarray, scale: int = 1) -> np.ndarray:
    """(2, 3, H, W) left/right pair -> (H*s, 2W*s, 3) side-by-side image (left on the left)."""
    left, right = to_hwc(stereo[0]), to_hwc(stereo[1])
    return upscale(np.concatenate([left, right], axis=1), scale)


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Red-cyan anaglyph: red channel from the left eye, green/blue from the right."""
    gl = left.astype(np.float32).mean(axis=-1)
    gr = right.astype(np.float32).mean(axis=-1)
    return np.stack([gl, gr, gr], axis=-1).clip(0, 255).astype(np.uint8)


def depth_to_rgb(depth: np.ndarray, near: float | None = None, far: float | None = None) -> np.ndarray:
    """Depth (H, W) metres -> grey image, near = bright. Non-finite pixels are black."""
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo = float(np.min(d[finite])) if near is None else near
    hi = float(np.max(d[finite])) if far is None else far
    g = np.zeros_like(d)
    g[finite] = 1.0 - (d[finite] - lo) / max(hi - lo, 1e-6)
    g = (g.clip(0, 1) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def draw_cross(img: np.ndarray, u: float, v: float, color, size: int = 3) -> np.ndarray:
    """Draw a small cross centred on continuous pixel coordinates (u right, v down)."""
    h, w = img.shape[:2]
    if not (np.isfinite(u) and np.isfinite(v)):
        return img
    cu, cv = int(np.floor(u)), int(np.floor(v))
    for k in range(-size, size + 1):
        if 0 <= cv < h and 0 <= cu + k < w:
            img[cv, cu + k] = color
        if 0 <= cv + k < h and 0 <= cu < w:
            img[cv + k, cu] = color
    return img


def save_png(img: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)


def save_mp4(frames: list[np.ndarray], path: str | Path, fps: float) -> None:
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for f in frames:
            writer.append_data(f)


def wandb_video(mp4_path: str | Path):
    """wandb.Video from an mp4 already written by save_mp4 (no moviepy needed)."""
    import wandb

    return wandb.Video(str(mp4_path), format="mp4")
