"""CPU tests for the two-arm needle handover: needle geometry, camera framing, and the scripted
handover controller driving an idealised version of the scene (tools go exactly where the relative
action says, a closed gripper next to the needle carries it)."""

import itertools
import math

import pytest
import torch

from config import load_config
from surgical_env.handover_geom import (
    NEEDLE_SCALE,
    arc_point_local,
    arc_points_local,
    grasp_yaw_deg,
    needle_radius,
    quat_about_z,
    update_handover_stages,
)
from surgical_env.rotations import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul
from surgical_env.scripted import B_CARRY, A_RETREAT, ScriptedNeedleHandover
from surgical_env.stereo import make_stereo_rig, project_points

CFG = "configs/needle_handover.yaml"


def test_needle_arc_geometry():
    r = needle_radius(NEEDLE_SCALE)
    assert r == pytest.approx(0.02)
    assert arc_point_local(180.0, NEEDLE_SCALE) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)  # root frame is on the needle
    assert arc_point_local(90.0, NEEDLE_SCALE) == pytest.approx((r, r, 0.0))  # the two ends
    assert arc_point_local(270.0, NEEDLE_SCALE) == pytest.approx((r, -r, 0.0))
    pts = arc_points_local(NEEDLE_SCALE)
    centre = torch.tensor([r, 0.0, 0.0])
    assert torch.allclose(torch.linalg.norm(pts - centre, dim=1), torch.full((pts.shape[0],), r, dtype=pts.dtype))
    # arm 2's grasp point is ~2 cm along the needle from arm 1's
    g = torch.tensor(arc_point_local(125.0, NEEDLE_SCALE))
    assert float(torch.linalg.norm(g)) == pytest.approx(2 * r * math.sin(math.radians(27.5)), rel=1e-6)


def test_grasp_yaw_turns_the_apex_wire_direction_into_the_grasp_point_one():
    def tangent(theta_deg):
        t = math.radians(theta_deg)
        return torch.tensor([-math.sin(t), math.cos(t), 0.0], dtype=torch.float64)

    for theta in (100.0, 125.0, 150.0, 200.0):
        q = torch.tensor(quat_about_z(grasp_yaw_deg(theta)), dtype=torch.float64)
        assert torch.allclose(quat_apply(q, tangent(180.0)), tangent(theta), atol=1e-12)


def test_handover_camera_frames_both_arms_needle_area_handover_point_and_goal():
    cfg = load_config(CFG)
    env, cam = cfg.env, cfg.camera
    assert env.task == "handover" and cfg.agent.proprio
    top = env.tissue_size[2]
    rig = make_stereo_rig(cam, (env.handover_look_at_x, 0.0, top + cam.look_at_height))
    bx = env.handover_base_x
    tips = [(-bx, 0.0, 0.08), (bx, 0.0, 0.08)]  # tool tips' start positions (~8 cm up, under each base)
    r = needle_radius(NEEDLE_SCALE)
    nx, xy = env.handover_needle_x, env.needle_xy_range
    pts = list(tips)
    pts += [(x, y, top) for x, y in itertools.product((-0.05, 0.05), (-0.05, 0.05))]  # the whole pad
    # every needle landing spot, with the whole half circle (it extends +2r in x and +-r in y from its root)
    pts += [(nx + dx + ex, dy + ey, top + 0.002) for dx, dy in itertools.product((-xy, xy), (-xy, xy))
            for ex, ey in ((0.0, 0.0), (r, r), (r, -r))]
    hp = (0.0, 0.0, top + env.goal_height)
    goal = (env.handover_goal_x, 0.0, top + env.goal_height)
    pts += [hp, goal, (goal[0] + 2 * r, r, goal[2]), (goal[0] + 2 * r, -r, goal[2])]
    k = torch.tensor([[rig.fx, 0, cam.width / 2], [0, rig.fx, cam.height / 2], [0, 0, 1]], dtype=torch.float64)
    us = []
    for cam_pos in (rig.left_pos, rig.right_pos):
        uv, z = project_points(torch.tensor([pts], dtype=torch.float64), torch.tensor([cam_pos], dtype=torch.float64),
                               torch.tensor([rig.quat_ros], dtype=torch.float64), k[None])
        assert (z > 0).all()
        assert (uv[..., 0] > 2).all() and (uv[..., 0] < cam.width - 2).all(), uv[..., 0]
        assert (uv[..., 1] > 2).all() and (uv[..., 1] < cam.height - 2).all(), uv[..., 1]
        us.append(uv[0, :2, 0])
    assert all(u[0] < u[1] for u in us)  # arm 1 on the left of the images, arm 2 on the right
    assert rig.left_pos[2] < 0.15 and rig.right_pos[2] < 0.15  # below the robot bases
    assert rig.left_pos[1] < 0 and rig.right_pos[1] < 0  # in front of the arms (they stand along x at y = 0)
    disparity = rig.expected_disparity(cam.distance)
    assert 2.0 < disparity < 5.0


class IdealHandoverScene:
    """Kinematic stand-in for the handover scene: tools go exactly where the relative IK action puts
    them, and a gripper that closes within 3 mm of the needle carries it rigidly until it opens."""

    def __init__(self, n, cfg, seed=0):
        g = torch.Generator().manual_seed(seed)
        e = cfg.env
        self.cfg, self.num_envs, self.device, self.step_dt = cfg, n, torch.device("cpu"), 0.04
        top = e.tissue_size[2]
        self.base = [torch.tensor([-e.handover_base_x, 0.0, 0.15]).repeat(n, 1),
                     torch.tensor([e.handover_base_x, 0.0, 0.15]).repeat(n, 1)]
        self.base_quat = torch.tensor([1.0, 0, 0, 0]).repeat(n, 1)
        self.tip = [b + torch.tensor([0.0, 0.0, -0.07]) for b in self.base]
        tilt = lambda: torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0]) + 0.1 * torch.randn(n, 4, generator=g), dim=1)
        self.quat = [tilt(), tilt()]
        jitter = (torch.rand(n, 2, generator=g) * 2 - 1) * e.needle_xy_range
        self.needle_pos = torch.stack([e.handover_needle_x + jitter[:, 0], jitter[:, 1], torch.full((n,), top + 0.0015)], 1)
        yaw = (torch.rand(n, generator=g) * 2 - 1) * 0.3
        self.needle_quat = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
        self.handover_point = torch.tensor([0.0, 0.0, top + e.goal_height]).repeat(n, 1)
        self.goal = torch.tensor([e.handover_goal_x, 0.0, top + e.goal_height]).repeat(n, 1)
        self.closed = [torch.zeros(n, dtype=torch.bool) for _ in range(2)]
        self.holder = torch.full((n,), -1)  # which arm carries the needle (-1: none)
        self.rel_pos = torch.zeros(n, 3)
        self.rel_quat = torch.tensor([1.0, 0, 0, 0]).repeat(n, 1)
        self.arc = arc_points_local(NEEDLE_SCALE)
        self.ever_held = [torch.zeros(n, dtype=torch.bool) for _ in range(2)]

    def handover_poses(self):
        return {"ee_pos": list(self.tip), "ee_quat": list(self.quat), "base_pos": self.base, "base_quat": [self.base_quat] * 2,
                "needle_pos": self.needle_pos, "needle_quat": self.needle_quat,
                "handover_point": self.handover_point, "goal": self.goal}

    def tip_needle_distance(self, tip):
        k = self.arc.shape[0]
        arc_w = self.needle_pos[:, None] + quat_apply(self.needle_quat[:, None].expand(-1, k, -1), self.arc[None].expand(self.num_envs, -1, -1))
        return torch.linalg.vector_norm(arc_w - tip[:, None], dim=-1).amin(1)

    def step(self, action):
        e = self.cfg.env
        for i in range(2):
            a = action[:, 7 * i:7 * i + 7]
            self.tip[i] = self.tip[i] + a[:, :3] * e.ik_pos_scale
            rv = a[:, 3:6] * e.ik_rot_scale
            ang = torch.linalg.vector_norm(rv, dim=1, keepdim=True)
            axis = rv / ang.clamp(min=1e-12)
            dq = torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], 1)
            self.quat[i] = quat_mul(dq, self.quat[i])
            closing = (a[:, 6] < 0) & ~self.closed[i]
            opening = (a[:, 6] >= 0) & self.closed[i]
            self.closed[i] = a[:, 6] < 0
            grab = closing & (self.tip_needle_distance(self.tip[i]) < 0.003)
            # rigid attachment: needle pose relative to the tool
            qc = quat_conjugate(self.quat[i])
            self.rel_pos = torch.where(grab[:, None], quat_apply(qc, self.needle_pos - self.tip[i]), self.rel_pos)
            self.rel_quat = torch.where(grab[:, None], quat_mul(qc, self.needle_quat), self.rel_quat)
            self.holder = torch.where(grab, torch.full_like(self.holder, i), self.holder)
            self.holder = torch.where(opening & (self.holder == i), torch.full_like(self.holder, -1), self.holder)
            self.ever_held[i] |= grab
        for i in range(2):
            m = self.holder == i
            self.needle_pos = torch.where(m[:, None], self.tip[i] + quat_apply(self.quat[i], self.rel_pos), self.needle_pos)
            self.needle_quat = torch.where(m[:, None], quat_mul(self.quat[i], self.rel_quat), self.needle_quat)


def test_scripted_handover_passes_the_needle_and_carries_it_to_the_goal():
    cfg = load_config(CFG)
    e = cfg.env
    n = 16
    scene = IdealHandoverScene(n, cfg)
    ctrl = ScriptedNeedleHandover(n, scene.step_dt, "cpu", e.ik_pos_scale, e.ik_rot_scale, NEEDLE_SCALE, e.handover_grasp_deg)
    ctrl.reset_idx()
    steps = int(e.episode_length_s / scene.step_dt)
    lift_h = e.tissue_size[2] + e.lift_height
    picked = torch.zeros(n, dtype=torch.bool)
    handed = torch.zeros(n, dtype=torch.bool)
    for _ in range(steps):
        a = ctrl.act(scene)
        assert a.shape == (n, 14) and (a.abs() <= 1.0).all()
        scene.step(a)
        # the environment's stage bookkeeping, on the same states
        lifted = scene.needle_pos[:, 2] > lift_h
        d1, d2 = scene.tip_needle_distance(scene.tip[0]), scene.tip_needle_distance(scene.tip[1])
        picked, handed, held_by_2 = update_handover_stages(picked, handed, lifted, d1, d2, e.hold_distance, e.release_distance)
    assert picked.all() and handed.all() and held_by_2.all()
    assert (ctrl.state_1 == A_RETREAT).all() and (ctrl.state_2 == B_CARRY).all()
    assert scene.ever_held[0].all() and scene.ever_held[1].all()  # each arm grasped the needle
    assert (scene.holder == 1).all()  # arm 2 has it at the end
    goal_dist = torch.linalg.vector_norm(scene.needle_pos - scene.goal, dim=1)
    assert (goal_dist < e.success_threshold).all(), goal_dist
    assert (scene.tip_needle_distance(scene.tip[0]) > e.release_distance).all()  # arm 1 is clear of it
    assert (scene.tip_needle_distance(scene.tip[1]) < e.hold_distance).all()


def test_handover_rotation_target_is_reachable_in_time():
    """Arm 2 has to turn its tool by the grasp yaw while it approaches; the stage timers must allow it."""
    cfg = load_config(CFG)
    e = cfg.env
    turn = math.radians(abs(grasp_yaw_deg(e.handover_grasp_deg)))
    steps_needed = turn / e.ik_rot_scale
    ctrl = ScriptedNeedleHandover(1, 0.04, "cpu", e.ik_pos_scale, e.ik_rot_scale, NEEDLE_SCALE, e.handover_grasp_deg)
    above_and_approach = float(ctrl.waits_2[1] + ctrl.waits_2[2]) / 0.04
    assert steps_needed < above_and_approach
    q = torch.tensor(quat_about_z(grasp_yaw_deg(e.handover_grasp_deg)))
    assert float(torch.linalg.vector_norm(axis_angle_from_quat(q))) == pytest.approx(turn, rel=1e-5)
