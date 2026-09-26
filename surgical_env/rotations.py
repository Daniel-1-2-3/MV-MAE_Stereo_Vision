"""Small quaternion / frame helpers in pure PyTorch.

Conventions match Isaac Lab (``isaaclab.utils.math``): quaternions are (w, x, y, z).
Kept free of Isaac imports so the scripted controller and the stereo geometry
can be unit-tested without the simulator.
"""

from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by unit quaternions q."""
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Rotation vector (axis * angle) of the shortest rotation represented by q."""
    q = q * (1.0 - 2.0 * (q[..., :1] < 0.0))
    mag = torch.linalg.norm(q[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = torch.where(angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48)
    return q[..., 1:] / sin_half_over_angle.unsqueeze(-1)


def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pose of frame 2 expressed in frame 1, given both in frame 0."""
    q10 = quat_conjugate(q01)
    t12 = quat_apply(q10, t02 - t01)
    q12 = quat_mul(q10, q02) if q02 is not None else None
    return t12, q12
