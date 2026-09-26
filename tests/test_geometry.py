"""CPU tests for the pure-math parts: rotations, stereo rig, scripted controller, config."""

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from config import CameraConfig, load_config
from surgical_env.rotations import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)
from surgical_env.scripted import APPROACH, APPROACH_ABOVE, GRASP, LIFT, REST, ScriptedNeedleLifter, relative_action
from surgical_env.stereo import look_at_rotation_ros, make_stereo_rig, project_points, quat_from_matrix


def rand_quat(n, seed=0):
    r = Rotation.random(n, random_state=seed)
    xyzw = r.as_quat()
    return torch.tensor(np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1), dtype=torch.float64), r


def test_quaternion_ops_match_scipy():
    q1, r1 = rand_quat(64, 1)
    q2, r2 = rand_quat(64, 2)
    v = torch.randn(64, 3, dtype=torch.float64)
    np.testing.assert_allclose(quat_apply(q1, v).numpy(), r1.apply(v.numpy()), atol=1e-10)
    prod = quat_mul(q1, q2)
    np.testing.assert_allclose(matrix_from_quat(prod).numpy(), (r1 * r2).as_matrix(), atol=1e-10)
    np.testing.assert_allclose(matrix_from_quat(q1).numpy(), r1.as_matrix(), atol=1e-10)
    np.testing.assert_allclose(matrix_from_quat(quat_conjugate(q1)).numpy(), r1.inv().as_matrix(), atol=1e-10)
    rv = axis_angle_from_quat(q1).numpy()
    np.testing.assert_allclose(Rotation.from_rotvec(rv).as_matrix(), r1.as_matrix(), atol=1e-8)
    assert np.all(np.linalg.norm(rv, axis=1) <= math.pi + 1e-9)  # shortest rotation


def test_subtract_frame_transforms():
    q01, r01 = rand_quat(16, 3)
    q02, r02 = rand_quat(16, 4)
    t01, t02 = torch.randn(16, 3, dtype=torch.float64), torch.randn(16, 3, dtype=torch.float64)
    t12, q12 = subtract_frame_transforms(t01, q01, t02, q02)
    np.testing.assert_allclose(t12.numpy(), r01.inv().apply((t02 - t01).numpy()), atol=1e-10)
    np.testing.assert_allclose(matrix_from_quat(q12).numpy(), (r01.inv() * r02).as_matrix(), atol=1e-10)


def test_look_at_and_quaternion():
    eye, target = np.array([0.1, 0.07, 0.12]), np.array([0.0, 0.0, 0.01])
    r = look_at_rotation_ros(eye, target)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    np.testing.assert_allclose(r[:, 2], fwd, atol=1e-12)
    assert r[2, 1] < 0  # camera "down" axis points down in the world
    assert abs(r[2, 0]) < 1e-12  # camera "right" axis is horizontal (no roll)
    q = quat_from_matrix(r)
    np.testing.assert_allclose(matrix_from_quat(torch.tensor(q)).numpy(), r, atol=1e-12)
    for seed in range(20):  # all branches of the matrix->quaternion conversion
        m = Rotation.random(random_state=seed).as_matrix()
        np.testing.assert_allclose(matrix_from_quat(torch.tensor(quat_from_matrix(m))).numpy(), m, atol=1e-10)


def test_stereo_rig_projection_and_disparity():
    cam = CameraConfig()
    look_at = (0.0, 0.0, 0.02)
    rig = make_stereo_rig(cam, look_at)
    left, right = np.array(rig.left_pos), np.array(rig.right_pos)
    assert np.linalg.norm(right - left) == pytest.approx(cam.baseline)
    q = torch.tensor([rig.quat_ros] * 2, dtype=torch.float64)
    pos = torch.tensor(np.stack([left, right]), dtype=torch.float64)
    k = torch.tensor([[rig.fx, 0, cam.width / 2], [0, rig.fx, cam.height / 2], [0, 0, 1]], dtype=torch.float64)
    k = k.expand(2, 3, 3)
    # the look-at point projects near the centre of both images, on the same row
    pts = torch.tensor([look_at], dtype=torch.float64)[None].expand(2, 1, 3)
    uv, z = project_points(pts, pos, q, k)
    assert abs(uv[0, 0, 1] - cam.height / 2) < 1e-6 and abs(uv[1, 0, 1] - cam.height / 2) < 1e-6
    disparity = uv[0, 0, 0] - uv[1, 0, 0]
    assert disparity > 0
    assert float(disparity) == pytest.approx(rig.expected_disparity(float(z[0, 0])), rel=1e-6)
    # a point to the camera's right lands to the right; a higher point lands higher in the image (smaller v)
    r = look_at_rotation_ros((left + right) / 2, look_at)
    for delta, axis, sign in ((r[:, 0] * 0.01, 0, +1), (np.array([0, 0, 0.01]), 1, -1)):
        p2 = torch.tensor(np.array([np.array(look_at) + delta]), dtype=torch.float64)[None].expand(2, 1, 3)
        uv2, _ = project_points(p2, pos, q, k)
        assert sign * (uv2[0, 0, axis] - uv[0, 0, axis]) > 0
    # 96 px, 24 mm focal length, 20.955 mm aperture, 5 mm baseline, 14 cm away: a few pixels of disparity
    assert 3.0 < rig.expected_disparity(cam.distance) < 5.0


def isaac_apply_delta_pose(pos, quat, delta):
    """Isaac Lab's isaaclab.utils.math.apply_delta_pose (relative IK command)."""
    angle = torch.linalg.norm(delta[:, 3:], dim=1)
    axis = delta[:, 3:] / angle[:, None]
    half = angle / 2
    dq = torch.cat([torch.cos(half)[:, None], torch.sin(half)[:, None] * axis], dim=1)
    dq = torch.where(angle[:, None] > 1e-6, dq, torch.tensor([1.0, 0, 0, 0], dtype=pos.dtype))
    return pos + delta[:, :3], quat_mul(dq, quat)


def test_relative_action_inverts_isaac_ik_command():
    n = 32
    ee_q, _ = rand_quat(n, 5)
    small, _ = rand_quat(n, 6)
    # desired orientation within a small rotation of the current one (inside the action range)
    rv = axis_angle_from_quat(small)
    rv = rv / torch.linalg.norm(rv, dim=1, keepdim=True) * 0.03
    des_q = quat_mul(torch.from_numpy(Rotation.from_rotvec(rv.numpy()).as_quat()[:, [3, 0, 1, 2]]), ee_q)
    ee_p = torch.randn(n, 3, dtype=torch.float64) * 0.05
    des_p = ee_p + torch.randn(n, 3, dtype=torch.float64) * 0.002
    grip = torch.ones(n, dtype=torch.float64)
    a = relative_action(ee_p, ee_q, des_p, des_q, grip, pos_scale=0.005, rot_scale=0.05)
    assert a.shape == (n, 7) and float(a.abs().max()) <= 1.0
    unclipped = (a[:, :6].abs() < 1.0).all(dim=1)
    assert unclipped.sum() > n // 2
    delta = a[:, :6] * torch.tensor([0.005] * 3 + [0.05] * 3, dtype=torch.float64)
    p, q = isaac_apply_delta_pose(ee_p, ee_q, delta)
    np.testing.assert_allclose(p[unclipped].numpy(), des_p[unclipped].numpy(), atol=1e-10)
    np.testing.assert_allclose(matrix_from_quat(q[unclipped]).numpy(), matrix_from_quat(des_q[unclipped]).numpy(), atol=1e-9)


def test_scripted_state_machine_timing():
    dt = 0.04
    sm = ScriptedNeedleLifter(2, dt, "cpu", 0.005, 0.05)
    z = torch.zeros(2, 3)
    q = torch.tensor([[1.0, 0, 0, 0]] * 2)
    states = []
    for _ in range(80):
        a = sm.compute(z, q, z + 0.01, q, z + 0.02, q)
        states.append(int(sm.state[0]))
        assert a.shape == (2, 7)
    # each state lasts ceil(wait / dt) + 1 steps, like the original warp kernel
    first = {s: states.index(s) for s in (APPROACH_ABOVE, APPROACH, GRASP, LIFT)}
    assert states[0] == REST
    assert first[APPROACH_ABOVE] == math.ceil(0.5 / dt - 1e-9)
    assert first[LIFT] > first[GRASP] > first[APPROACH] > first[APPROACH_ABOVE]
    assert states[-1] == LIFT
    sm.reset_idx(torch.tensor([1]))
    assert int(sm.state[1]) == REST and int(sm.state[0]) == LIFT


def test_config_loading(tmp_path):
    cfg = load_config("configs/needle_tissue.yaml", ["env.num_envs=4", "agent.lr=3e-4", "camera.baseline=0.008",
                                                    "env.tissue_size=[0.1,0.1,0.02]", "train.total_env_steps=1e6"])
    assert cfg.env.num_envs == 4 and cfg.agent.lr == pytest.approx(3e-4) and cfg.camera.baseline == 0.008
    assert cfg.env.tissue_size == (0.1, 0.1, 0.02) and cfg.train.total_env_steps == 1_000_000
    assert cfg.env.tissue_youngs_modulus == pytest.approx(5e4)
    with pytest.raises(KeyError):
        load_config(None, ["env.not_a_key=1"])
    with pytest.raises(TypeError):
        load_config(None, ["env.num_envs=abc"])
    with pytest.raises(ValueError):
        load_config(None, ["mvmae.patch_size=12"])
    again = load_config(overrides=["env.num_envs=2"], base=cfg.to_dict())
    assert again.env.num_envs == 2 and again.agent.lr == pytest.approx(3e-4)


def test_default_framing_shows_tool_needle_area_and_goal():
    """The default rig must see the tool tip's start (~8-10 cm up), the goal and every needle spawn position."""
    import itertools

    from config import Config

    cfg = Config()
    cam, env = cfg.camera, cfg.env
    top = env.tissue_size[2]
    rig = make_stereo_rig(cam, (env.goal_xy[0], env.goal_xy[1], top + cam.look_at_height))
    r = env.needle_xy_range + 0.005
    pts = [(0.0, 0.0, 0.08), (0.0, 0.0, 0.10), (env.goal_xy[0], env.goal_xy[1], top + env.goal_height)]
    pts += [(x, y, top + 0.001) for x, y in itertools.product((-r, r), (-r, r))]
    k = torch.tensor([[rig.fx, 0, cam.width / 2], [0, rig.fx, cam.height / 2], [0, 0, 1]], dtype=torch.float64)
    for cam_pos in (rig.left_pos, rig.right_pos):
        uv, z = project_points(torch.tensor([pts], dtype=torch.float64), torch.tensor([cam_pos], dtype=torch.float64),
                               torch.tensor([rig.quat_ros], dtype=torch.float64), k[None])
        assert (z > 0).all()
        assert (uv[..., 0] > 3).all() and (uv[..., 0] < cam.width - 3).all()
        assert (uv[..., 1] > 3).all() and (uv[..., 1] < cam.height - 3).all()
    assert rig.left_pos[2] < 0.15 and rig.right_pos[2] < 0.15  # below the PSM base
