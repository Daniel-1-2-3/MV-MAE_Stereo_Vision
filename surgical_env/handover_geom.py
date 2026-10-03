"""Needle geometry and layout for the two-arm needle handover task.

No Isaac imports, so it can be unit-tested without the simulator.

The suture needle asset (Props/SutureNeedle/needle*.usd, spawned at scale 0.4) is a flat
half circle in its own x-y plane. Its root frame sits ON the needle, at the middle of the
arc (the "apex"); the arc's centre is at +radius along the needle's x axis and the two ends
at (radius, +-radius). The single-arm task grasps at the root frame with the tool turned
like the needle, so arm 1 does the same. Arm 2 grasps at another point of the arc, at
`theta` degrees around the arc centre (180 = the apex), with the tool turned by the same
angle as the needle's wire there.
"""

from __future__ import annotations

import math

import torch

NEEDLE_SCALE = 0.4  # the scale the original tasks spawn the needle asset at (a half circle of 2 cm radius)
NEEDLE_RADIUS_UNSCALED = 0.05  # metres in the USD file (arc from (0, -0.05) via (-0.05, 0) to (0, 0.05) around its centre)


def needle_radius(scale: float) -> float:
    return NEEDLE_RADIUS_UNSCALED * scale


def arc_point_local(theta_deg: float, scale: float) -> tuple[float, float, float]:
    """Point of the needle at angle `theta_deg` around the arc centre, in the needle's root frame."""
    r = needle_radius(scale)
    t = math.radians(theta_deg)
    return (r + r * math.cos(t), r * math.sin(t), 0.0)


def arc_points_local(scale: float, n: int = 25) -> torch.Tensor:
    """(n, 3) points spread along the whole half circle (angles 90..270 deg, 7.5 deg = 2.6 mm apart at n = 25), needle root frame."""
    return torch.tensor([arc_point_local(90.0 + 180.0 * i / (n - 1), scale) for i in range(n)])


def grasp_yaw_deg(theta_deg: float) -> float:
    """Rotation about the needle's z axis that turns the wire direction at the apex into the one at `theta_deg`."""
    return theta_deg - 180.0


def quat_about_z(angle_deg: float) -> tuple[float, float, float, float]:
    h = math.radians(angle_deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def update_handover_stages(
    picked: torch.Tensor,
    handed_over: torch.Tensor,
    lifted: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    hold_distance: float,
    release_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latch the handover stages for one step. d1 / d2: each tool tip's distance to the closest point of the needle.

    picked:      arm 1 holds the lifted needle while arm 2 is clear of it (arm 2 cannot take it off the pad itself)
    handed_over: after that, arm 2 holds the lifted needle while arm 1 is clear of it
    held_by_2:   (not latched) handed over, and right now lifted with arm 1 clear of it
    Returns (picked, handed_over, held_by_2).
    """
    picked = picked | (lifted & (d1 < hold_distance) & (d2 > release_distance))
    handed_over = handed_over | (picked & lifted & (d2 < hold_distance) & (d1 > release_distance))
    held_by_2 = handed_over & lifted & (d1 > release_distance)
    return picked, handed_over, held_by_2
