"""Stereo camera rig geometry.

The rig is two identical pinhole cameras with parallel optical axes, separated by
``baseline`` along the cameras' shared "right" axis -- the same layout as a
stereo endoscope. Poses use the ROS camera convention (x right, y down,
z forward), which Isaac Lab accepts via ``OffsetCfg(convention="ros")``.

No Isaac imports here, so everything can be unit-tested without the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .rotations import matrix_from_quat


def look_at_rotation_ros(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation matrix whose columns are the camera x (right), y (down), z (forward) axes in world."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        raise ValueError("camera looks straight along the up axis; choose a lower elevation")
    right /= norm
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix, with w >= 0."""
    m = np.asarray(r, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def focal_length_pixels(width: int, focal_length_mm: float, horizontal_aperture_mm: float) -> float:
    """Same formula Isaac Lab uses for Camera.data.intrinsic_matrices (square pixels)."""
    return width * focal_length_mm / horizontal_aperture_mm


@dataclass
class StereoRig:
    left_pos: tuple[float, float, float]  # relative to the environment origin
    right_pos: tuple[float, float, float]
    quat_ros: tuple[float, float, float, float]  # shared orientation, (w, x, y, z), ROS convention
    look_at: tuple[float, float, float]
    baseline: float
    fx: float  # focal length in pixels

    def expected_disparity(self, depth: float) -> float:
        """Horizontal pixel shift of a point at `depth` metres between left and right images."""
        return self.fx * self.baseline / depth


def make_stereo_rig(cam, look_at) -> StereoRig:
    """Place the rig from a CameraConfig and a look-at point (env-local coordinates)."""
    az, el = math.radians(cam.azimuth_deg), math.radians(cam.elevation_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    look_at = np.asarray(look_at, dtype=np.float64)
    center = look_at + cam.distance * direction
    rot = look_at_rotation_ros(center, look_at)
    right_axis = rot[:, 0]
    left = center - 0.5 * cam.baseline * right_axis
    right = center + 0.5 * cam.baseline * right_axis
    return StereoRig(
        left_pos=tuple(float(v) for v in left),
        right_pos=tuple(float(v) for v in right),
        quat_ros=tuple(float(v) for v in quat_from_matrix(rot)),
        look_at=tuple(float(v) for v in look_at),
        baseline=float(cam.baseline),
        fx=focal_length_pixels(cam.width, cam.focal_length, cam.horizontal_aperture),
    )


def project_points(
    points_w: torch.Tensor, cam_pos_w: torch.Tensor, cam_quat_ros: torch.Tensor, intrinsics: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinhole projection.

    Args:
        points_w: (N, K, 3) world points.
        cam_pos_w: (N, 3) camera positions.
        cam_quat_ros: (N, 4) camera orientations, ROS convention, (w, x, y, z).
        intrinsics: (N, 3, 3) camera matrices.

    Returns:
        uv: (N, K, 2) pixel coordinates (u to the right, v down).
        depth: (N, K) distance along the optical axis.
    """
    rot = matrix_from_quat(cam_quat_ros)  # camera -> world
    p_cam = torch.einsum("nji,nkj->nki", rot, points_w - cam_pos_w[:, None, :])  # R^T (p - t)
    depth = p_cam[..., 2]
    fx, fy = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None]
    cx, cy = intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]
    u = fx * p_cam[..., 0] / depth + cx
    v = fy * p_cam[..., 1] / depth + cy
    return torch.stack([u, v], dim=-1), depth
