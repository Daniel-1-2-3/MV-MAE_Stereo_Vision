"""Check that the stereo camera rig and the needle-on-tissue scene are set up correctly.

Run on a machine with an RTX GPU and Isaac Sim installed (see README):

    python test_stereo.py                      # headless, 4 environments
    python test_stereo.py --gui                # watch it in the Isaac Sim window
    python test_stereo.py camera.baseline=0.008 env.num_envs=2   # any config override

What it checks (each prints PASS / FAIL):
  Images      - right shape/dtype, not blank, left and right similar but not identical
  Rig         - camera positions/orientations reported by the simulator match the
                design: 5 mm apart, parallel, right camera on the right
  Projection  - needle and tool tip are inside both images, on the same pixel row
                in both, shifted sideways by the amount the geometry predicts,
                and not hidden behind something
  Stereo      - image-only check: warping the right image by the disparity that
                the depth map predicts reproduces the left image better than no
                shift, the opposite shift, or a vertical shift. A diagnostic then
                measures the whole-pixel shift that best aligns right to left
                (whole image and needle / tool region) next to the disparity the
                geometry predicts, checks the depth image against the pad's
                geometry, prints the renderer / camera settings, and says whether
                the images or only the prediction are off
  Tissue      - the needle rests on the tissue (not sunk through, not bouncing).
                Rigid pad (default): the tool pressed into it stays on its surface
                without a physics glitch. Soft pad (env.tissue_deformable=true): the
                pinned bottom stays put, the pad dents under the tool, and does not slide

Outputs (default outputs/stereo_check/):
  stereo_check.png  - per env: left | right | red-cyan anaglyph | left depth,
                      needle marked yellow, tool tip cyan
  stereo_check.mp4  - env 0 through the whole test, left | right
  results.json      - every number behind the checks
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/needle_tissue.yaml")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", default="outputs/stereo_check")
parser.add_argument("--settle_steps", type=int, default=50, help="control steps (25 per second) to let the needle and tissue settle")
simulation_app, args, overrides = launch(parser)

# ---- everything below needs the running simulator
import importlib.metadata  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import vis  # noqa: E402
from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from surgical_env.psm import PSM_BASE_POS  # noqa: E402
from surgical_env.rotations import matrix_from_quat, wxyz_from_xyzw  # noqa: E402
from surgical_env.scripted import GRIPPER_CLOSE, GRIPPER_OPEN, relative_action  # noqa: E402
from surgical_env.stereo import project_points  # noqa: E402

results: list[dict] = []


def check(name: str, passed: bool, detail: str, **numbers) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail, **numbers})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def f(x) -> list[float]:
    return [round(float(v), 5) for v in torch.as_tensor(x).flatten().tolist()]


def shift_errors(left: torch.Tensor, right: torch.Tensor, max_dx: int = 10, max_dy: int = 0,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """Photometric error of aligning the right image to the left by whole-pixel shifts.

    left, right: (n, 3, H, W) float. Returns (n, 2 * max_dy + 1, 2 * max_dx + 1): entry [dy, dx] is the mean
    |left(u, v) - right(u - dx, v - dy)| over the overlap (and `mask`, (n, H, W) bool, if given). A point the
    left camera sees at column u is at u - d in the right image (right camera on the right), so for a working
    stereo pair the best dx is the disparity d > 0 and the best dy is 0.
    """
    n, _, hh, ww = left.shape
    err = torch.full((n, 2 * max_dy + 1, 2 * max_dx + 1), float("inf"), device=left.device)
    for iy, dy in enumerate(range(-max_dy, max_dy + 1)):
        for ix, dx in enumerate(range(-max_dx, max_dx + 1)):
            u0, u1, v0, v1 = max(dx, 0), ww + min(dx, 0), max(dy, 0), hh + min(dy, 0)
            diff = (left[:, :, v0:v1, u0:u1] - right[:, :, v0 - dy:v1 - dy, u0 - dx:u1 - dx]).abs().mean(dim=1)
            if mask is None:
                err[:, iy, ix] = diff.mean(dim=(1, 2))
            else:
                m = mask[:, v0:v1, u0:u1].float()
                cnt = m.sum(dim=(1, 2))
                err[:, iy, ix] = torch.where(cnt > 0, (diff * m).sum(dim=(1, 2)) / cnt.clamp(min=1),
                                             torch.full_like(cnt, float("inf")))
    return err


def subpixel_best(err_row: torch.Tensor, offset: int) -> torch.Tensor:
    """(n, k) errors over shifts -offset..offset -> best shift per row, refined with a parabola through the minimum."""
    i = err_row.argmin(dim=1)
    best = (i - offset).float()
    k = err_row.shape[1]
    for r in range(err_row.shape[0]):
        j = int(i[r])
        if 0 < j < k - 1:
            a, b, c = (float(err_row[r, j + s]) for s in (-1, 0, 1))
            den = a - 2 * b + c
            if den > 1e-9:
                best[r] += 0.5 * (a - c) / den
    return best


def stereo_diagnostic(env, obs, depth, uv_l, uv_r, z_l, pl, pr, ql, rot_l, K, disp_map, origins, tissue_top_local,
                      needle_w, rig) -> None:
    """Did the images lose their stereo disparity, or did only the depth-based prediction change?

    Measures the whole-pixel horizontal shift that best aligns the right image to the left (whole image and the
    needle / tool-tip region) and prints it next to the disparity the geometry predicts; checks that the depth
    image agrees with the camera geometry on the pad (catches a renderer projection / field of view that differs
    from the intrinsics); prints the camera prims' optics, the depth output, and the right camera's offset in the
    left camera's frame. Ends with the verdict.
    """
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    n, _, _, h, w = obs.shape
    dev = obs.device
    left, right = obs[:, 0].float(), obs[:, 1].float()
    print("-" * 72 + "\n[stereo diagnostic]", flush=True)

    # ---- renderer, optics and outputs as the simulator has them (2: depth type / units / shape, intrinsics)
    sim = env.env.sim
    print("  renderer: " + ", ".join(f"{k}={sim.get_setting(k)}" for k in (
        "/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx-transient/dldenoiser/enabled")))
    stage = sim_utils.get_current_stage()
    env0 = env.env.scene.env_prim_paths[0]
    for name in ("StereoLeft", "StereoRight"):
        cam = UsdGeom.Camera(stage.GetPrimAtPath(f"{env0}/{name}"))
        proj = cam.GetPrim().GetAttribute("cameraProjectionType")
        print(f"  {name} prim: focal length {cam.GetFocalLengthAttr().Get()}, horizontal aperture "
              f"{cam.GetHorizontalApertureAttr().Get()}, vertical aperture {cam.GetVerticalApertureAttr().Get()}, "
              f"clipping {cam.GetClippingRangeAttr().Get()}, projection {proj.Get() if proj else 'n/a'}")
    print(f"  intrinsics in use (left, env 0): fx {float(K[0, 0, 0]):.3f}, fy {float(K[0, 1, 1]):.3f}, "
          f"cx {float(K[0, 0, 2]):.2f}, cy {float(K[0, 1, 2]):.2f} for {w} x {h} px; design fx {rig.fx:.3f}")
    outs = {k: (tuple(v.torch.shape), str(v.torch.dtype)) for k, v in env.left.data.output.items()}
    print(f"  left camera data_types {list(env.left.cfg.data_types)} -> outputs {outs}")
    d0 = depth[0, 0]
    fin = torch.isfinite(d0) & (d0 > 0)
    if bool(fin.any()):
        print(f"  depth image (distance_to_image_plane, env 0 left): {float(fin.float().mean()):.2f} of pixels finite, "
              f"min / median / max {float(d0[fin].min()):.4f} / {float(d0[fin].median()):.4f} / {float(d0[fin].max()):.4f} m")

    # ---- the two cameras' poses (3: right camera 5 mm along the rig's right axis, same orientation)
    off = torch.einsum("nji,nj->ni", rot_l, pr - pl) * 1000.0  # right - left position in the left camera frame (x right)
    for i in range(n):
        print(f"  env {i}: right camera offset in the left camera's frame (x right, y down, z forward) "
              f"{[round(float(v), 3) for v in off[i]]} mm, design [{rig.baseline * 1000:.1f}, 0, 0]")

    # ---- 1: measured shift vs geometry
    lr = (left - right).abs()
    roi = torch.zeros(n, h, w, dtype=torch.bool, device=dev)  # +-8 px around the projected needle and tool tip
    for i in range(n):
        for j in range(uv_l.shape[1]):
            u, v = float(uv_l[i, j, 0]), float(uv_l[i, j, 1])
            if np.isfinite(u) and np.isfinite(v):
                u0, u1, v0, v1 = max(0, int(u) - 8), min(w, int(u) + 9), max(0, int(v) - 8), min(h, int(v) + 9)
                if u1 > u0 and v1 > v0:
                    roi[i, v0:v1, u0:u1] = True
    err_whole = shift_errors(left, right, 10, 0)[:, 0]
    err_roi = shift_errors(left, right, 10, 0, roi)[:, 0]
    err_2d = shift_errors(left, right, 10, 3)
    best_whole, best_roi = subpixel_best(err_whole, 10), subpixel_best(err_roi, 10)
    flat = err_2d.flatten(1).argmin(dim=1)
    best_dy, best_dx_2d = flat // 21 - 3, flat % 21 - 10
    pred_geom = rig.fx * rig.baseline / z_l  # (n, 2): needle, tool tip
    proj_disp = uv_l[..., 0] - uv_r[..., 0]
    pred_map = torch.stack([disp_map[i][disp_map[i] > 0].median() if bool((disp_map[i] > 0).any())
                            else torch.tensor(float("nan"), device=dev) for i in range(n)])

    # ---- depth image vs geometry on the pad top (renderer projection = intrinsics?)
    g = torch.linspace(-0.04, 0.04, 9, device=dev)
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (81, 2) pad-local x, y
    top = origins[:, 2] + tissue_top_local
    pts = torch.cat([origins[:, None, :2] + grid[None], top[:, None, None].expand(n, grid.shape[0], 1)], dim=-1)
    far_from_needle = torch.linalg.norm(pts[..., :2] - needle_w[:, None, :2], dim=-1) > 0.015
    uv_p, z_p = project_points(pts, pl, ql, K)
    ui, vi = uv_p[..., 0].floor().long(), uv_p[..., 1].floor().long()
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z_p > 0) & far_from_needle
    d_img = depth[torch.arange(n, device=dev)[:, None], 0, vi.clamp(0, h - 1), ui.clamp(0, w - 1)]
    depth_err_mm = torch.stack([((d_img[i] - z_p[i])[inside[i]]).abs().median() if bool(inside[i].any())
                                else torch.tensor(float("nan"), device=dev) for i in range(n)]) * 1000.0
    depth_ok = bool((depth_err_mm < 2.0).all())
    check("projection/depth_matches_geometry", depth_ok,
          f"median |depth image - geometric depth| at {int(inside.sum())} pad-top points {f(depth_err_mm)} mm "
          f"(want < 2: the rendered projection matches the intrinsics)", depth_err_mm=f(depth_err_mm))

    verdicts = []
    for i in range(n):
        p = float(pred_map[i])
        mw, mr = float(best_whole[i]), float(best_roi[i])
        print(f"  env {i}: best right->left shift: whole image {mw:+.2f} px (error {float(err_whole[i].min()):.2f}, "
              f"at 0 px {float(err_whole[i, 10]):.2f}), needle/tool region {mr:+.2f} px, 2-D search (dx, dy) = "
              f"({int(best_dx_2d[i])}, {int(best_dy[i])}) | geometry predicts: needle {float(pred_geom[i, 0]):.2f}, "
              f"tool {float(pred_geom[i, 1]):.2f} (projected {float(proj_disp[i, 0]):.2f}, {float(proj_disp[i, 1]):.2f}), "
              f"depth-map median {p:.2f} px | mean |L-R| {float(lr[i].mean()):.2f}, max {float(lr[i].max()):.0f}")
        if abs(mw - p) <= max(1.0, 0.3 * p):
            verdicts.append("ok")
        elif abs(mw) < 0.75:
            verdicts.append("no_disparity")
        else:
            verdicts.append("different")
    measured_ok = all(v == "ok" for v in verdicts)
    check("stereo/measured_disparity", measured_ok,
          f"best whole-image shift {f(best_whole)} px vs depth-predicted {f(pred_map)} px "
          f"(needle / tool region {f(best_roi)} px; geometry {f(pred_geom[:, 0])} / {f(pred_geom[:, 1])} px)",
          measured_px=f(best_whole), measured_roi_px=f(best_roi), predicted_px=f(pred_map))
    if measured_ok and depth_ok:
        verdict = "the IMAGES have the stereo disparity the geometry predicts -> only the depth-based prediction is off"
    elif "no_disparity" in verdicts:
        verdict = "the IMAGES have (almost) no horizontal disparity -> the stereo pair is broken (camera setup)"
    elif not depth_ok:
        verdict = "the depth image does not match the camera geometry -> rendered projection differs from the intrinsics"
    else:
        verdict = "the images' disparity differs from the geometry's prediction (see the numbers above)"
    print(f"  VERDICT: {verdict}\n" + "-" * 72, flush=True)


def main() -> int:
    cfg = load_config(args.config, overrides + [f"env.num_envs={args.num_envs}"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pkg in ("isaacsim", "isaaclab", "torch"):
        try:
            print(f"{pkg} {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg} (version unknown)")

    env = make_env(cfg, device=args.device, with_depth=True, track_camera_pose=True)
    scene = env.env.scene
    n, dev = env.num_envs, env.device
    h, w = cfg.camera.height, cfg.camera.width
    origins = scene.env_origins
    needle, ee_frame = scene["object"], scene["ee_frame"]
    soft = cfg.env.tissue_deformable
    tissue = scene["tissue"] if soft else None
    tissue_top_local = cfg.env.tissue_size[2]
    rig = env.rig
    fps = 1.0 / env.step_dt
    video: list[np.ndarray] = []
    terminated_any = torch.zeros(n, dtype=torch.bool, device=dev)

    def step(action):
        nonlocal terminated_any
        obs, _, term, trunc, _ = env.step(action)
        terminated_any |= term | trunc
        video.append(vis.stereo_frame(obs[0], scale=3))
        return obs

    obs = env.reset()
    video.append(vis.stereo_frame(obs[0], scale=3))

    # ------------------------------------------------------------------ settle
    hold = torch.zeros(n, env.action_dim, device=dev)
    hold[:, 6] = GRIPPER_OPEN
    needle_pos_hist = []
    for _ in range(args.settle_steps):
        obs = step(hold)
        needle_pos_hist.append(needle.data.root_pos_w.torch.clone())

    # ------------------------------------------------------------------ images
    ok_shape = obs.shape == (n, 2, 3, h, w) and obs.dtype == torch.uint8
    check("images/shape", ok_shape, f"obs {tuple(obs.shape)} {obs.dtype}, expected ({n}, 2, 3, {h}, {w}) uint8")
    img = obs.float()
    std = img.flatten(2).std(dim=-1)  # (n, 2)
    mean = img.flatten(2).mean(dim=-1)
    not_blank = bool(((std > 2.0) & (mean > 5.0) & (mean < 250.0)).all())
    check("images/not_blank", not_blank, f"per-image std min {std.min():.1f}, mean range [{mean.min():.1f}, {mean.max():.1f}]",
          std=f(std), mean=f(mean))
    lr_diff = (img[:, 0] - img[:, 1]).abs().mean(dim=(1, 2, 3))
    similar = bool(((lr_diff > 0.2) & (lr_diff < 40.0)).all())
    check("images/left_vs_right", similar,
          f"mean |L-R| per env {f(lr_diff)} (want between 0.2 and 40: similar but not identical)", lr_diff=f(lr_diff))

    # ------------------------------------------------------------------ rig
    # (Isaac Lab 3.0: .torch views of the camera data; its quaternions are (x, y, z, w), ours (w, x, y, z))
    pl, pr = env.left.data.pos_w.torch, env.right.data.pos_w.torch
    ql, qr = wxyz_from_xyzw(env.left.data.quat_w_ros.torch), wxyz_from_xyzw(env.right.data.quat_w_ros.torch)
    K = env.left.data.intrinsic_matrices.torch
    baseline_m = torch.linalg.norm(pr - pl, dim=-1)
    check("rig/baseline", bool(((baseline_m - rig.baseline).abs() < 0.05 * rig.baseline).all()),
          f"measured {f(baseline_m * 1000)} mm, designed {rig.baseline * 1000:.2f} mm", baseline_m=f(baseline_m))
    dot = (ql * qr).sum(-1).abs().clamp(max=1.0)
    ang = torch.rad2deg(2 * torch.acos(dot))
    check("rig/parallel", bool((ang < 0.2).all()), f"angle between the two cameras {f(ang)} deg (want < 0.2)")
    q_design = torch.tensor(rig.quat_ros, device=dev).expand_as(ql)
    ang_design = torch.rad2deg(2 * torch.acos((ql * q_design).sum(-1).abs().clamp(max=1.0)))
    check("rig/orientation", bool((ang_design < 0.2).all()),
          f"left camera orientation vs design {f(ang_design)} deg (want < 0.2)")
    rot_l = matrix_from_quat(ql)
    right_dir = (pr - pl) / baseline_m[:, None]
    along_x = (rot_l[:, :, 0] * right_dir).sum(-1)
    check("rig/right_is_right", bool((along_x > 0.999).all()),
          f"cos(angle) between left camera's x axis and left->right offset {f(along_x)} (want > 0.999)")
    exp_l = torch.tensor(rig.left_pos, device=dev) + origins
    pos_err = torch.linalg.norm(pl - exp_l, dim=-1)
    check("rig/placement", bool((pos_err < 1e-3).all()), f"left camera position error {f(pos_err * 1000)} mm (want < 1)")
    fx_err = (K[:, 0, 0] - rig.fx).abs() / rig.fx
    check("rig/intrinsics", bool((fx_err < 1e-3).all()), f"fx {f(K[:, 0, 0])} px, designed {rig.fx:.3f} px")

    # ------------------------------------------------------------------ projection
    needle_w = needle.data.root_pos_w.torch
    tip_w = ee_frame.data.target_pos_w.torch[:, 0, :]
    pts = torch.stack([needle_w, tip_w], dim=1)  # (n, 2, 3)
    uv_l, z_l = project_points(pts, pl, ql, K)
    uv_r, z_r = project_points(pts, pr, qr, env.right.data.intrinsic_matrices.torch)
    depth = env.depths()  # (n, 2, h, w)
    for j, name in enumerate(("needle", "tool_tip")):
        inside = ((uv_l[:, j, 0] >= 0) & (uv_l[:, j, 0] < w) & (uv_l[:, j, 1] >= 0) & (uv_l[:, j, 1] < h)
                  & (uv_r[:, j, 0] >= 0) & (uv_r[:, j, 0] < w) & (uv_r[:, j, 1] >= 0) & (uv_r[:, j, 1] < h)
                  & (z_l[:, j] > 0) & (z_r[:, j] > 0))
        check(f"projection/{name}_in_view", bool(inside.all()),
              f"left uv {[f(x) for x in uv_l[:, j]]}, right uv {[f(x) for x in uv_r[:, j]]}")
        row_err = (uv_l[:, j, 1] - uv_r[:, j, 1]).abs()
        check(f"projection/{name}_same_row", bool((row_err < 0.5).all()), f"|v_left - v_right| {f(row_err)} px (want < 0.5)")
        disp = uv_l[:, j, 0] - uv_r[:, j, 0]
        expected = rig.fx * rig.baseline / z_l[:, j]
        ok = (disp > 0) & ((disp - expected).abs() < 0.05 * expected + 0.05)
        check(f"projection/{name}_disparity", bool(ok.all()),
              f"u_left - u_right {f(disp)} px, expected fx*B/Z {f(expected)} px", disparity=f(disp), expected=f(expected))
        # Not hidden: the depth image at the projected pixel is not much closer than the point.
        ui = uv_l[:, j, 0].floor().long().clamp(0, w - 1)
        vi = uv_l[:, j, 1].floor().long().clamp(0, h - 1)
        d_at = depth[torch.arange(n, device=dev), 0, vi, ui]
        visible = torch.isfinite(d_at) & (d_at > z_l[:, j] - 0.01)
        check(f"projection/{name}_not_occluded", bool((visible & inside).all()),
              f"depth image at the point {f(d_at)} m vs point depth {f(z_l[:, j])} m")

    # ------------------------------------------------------------------ image-only stereo consistency
    left_img, right_img = obs[:, 0].float(), obs[:, 1].float()
    dl = depth[:, 0]
    disp_map = torch.where(torch.isfinite(dl) & (dl > 0), rig.fx * rig.baseline / dl, torch.zeros_like(dl))
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")

    def warp_error(dx_sign: float, dy: float) -> torch.Tensor:
        src_x = xs[None] + 0.5 + dx_sign * disp_map  # continuous coordinate in the right image
        src_y = ys[None] + 0.5 + dy
        grid = torch.stack([2 * src_x / w - 1, 2 * src_y.expand_as(src_x) / h - 1], dim=-1)
        warped = F.grid_sample(right_img, grid, mode="bilinear", padding_mode="border", align_corners=False)
        valid = (src_x > 1) & (src_x < w - 1) & (src_y > 1) & (src_y < h - 1) & (disp_map > 0)
        err = (warped - left_img).abs().mean(dim=1)
        return (err * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)

    e_true = warp_error(-1.0, 0.0)  # left pixel u sees the point the right camera has at u - d
    e_zero = warp_error(0.0, 0.0)
    e_flip = warp_error(+1.0, 0.0)
    e_vert = warp_error(-1.0, 3.0)
    ok = (e_true < e_flip) & (e_true < e_vert) & (e_true <= 1.02 * e_zero)
    check("stereo/depth_warp", bool(ok.all()),
          f"photometric error: correct shift {f(e_true)}, no shift {f(e_zero)}, opposite shift {f(e_flip)}, "
          f"vertical offset {f(e_vert)}", e_true=f(e_true), e_zero=f(e_zero), e_flip=f(e_flip), e_vert=f(e_vert))
    stereo_diagnostic(env, obs, depth, uv_l, uv_r, z_l, pl, pr, ql, rot_l, K, disp_map, origins, tissue_top_local,
                      needle_w, rig)

    snapshot, snapshot_depth = obs.clone(), depth.clone()  # for the output image

    # ------------------------------------------------------------------ needle on tissue
    if soft:
        nodes = tissue.data.nodal_pos_w.torch  # (n, V, 3)
        rest = tissue.data.default_nodal_state_w.torch[..., :3]
        top_now = nodes[..., 2].max(dim=1).values
    else:
        top_now = origins[:, 2] + tissue_top_local
    nz = needle_w[:, 2]
    check("tissue/no_episode_end", not bool(terminated_any.any()),
          "no environment terminated or timed out during settling (a fall-through would end the episode)")
    on_top = (nz > top_now - 0.004) & (nz < top_now + 0.01)
    check("tissue/needle_on_surface", bool(on_top.all()),
          f"needle z {f(nz * 1000)} mm, tissue top {f(top_now * 1000)} mm (want within -4..+10 mm)")
    # Settled = the needle's position barely changes over the last 0.4 s. (Instantaneous
    # velocity is not used: contact with a soft body leaves sub-millimetre jitter.)
    hist = torch.stack(needle_pos_hist[-10:], dim=0)  # (10, n, 3)
    drift = torch.linalg.norm(hist - hist[-1:], dim=-1).max(dim=0).values
    z_drift = hist[..., 2].max(0).values - hist[..., 2].min(0).values
    speed = torch.linalg.norm(needle.data.root_lin_vel_w.torch, dim=-1)
    check("tissue/needle_settled", bool((drift < 1e-3).all()),
          f"needle movement over the last 10 steps {f(drift * 1000)} mm (want < 1), height change {f(z_drift * 1000)} mm, "
          f"instantaneous speed {f(speed)} m/s", drift_m=f(drift))
    if soft:
        rest_z = rest[..., 2]
        bottom = rest_z <= rest_z.min(dim=1, keepdim=True).values + 1e-4
    if soft and cfg.env.pin_tissue_bottom:
        pin_err = torch.linalg.norm(nodes - rest, dim=-1).masked_fill(~bottom, 0.0).max(dim=1).values
        check("tissue/bottom_pinned", bool((pin_err < 5e-4).all()), f"max drift of pinned bottom nodes {f(pin_err * 1000)} mm")

    # ------------------------------------------------------------------ press test
    local_needle = needle_w - origins
    sign = torch.where(local_needle[:, :2] >= 0, 1.0, -1.0)
    press_xy = -0.02 * sign  # 2 cm on the far side of the centre from the needle
    rest_top = (rest_z.max(dim=1).values - origins[:, 2]) if soft else torch.full((n,), tissue_top_local, device=dev)
    ended_before_press = terminated_any.clone()
    base_pos = torch.tensor(PSM_BASE_POS, device=dev)
    ee_pos, ee_quat, *_ = env.poses_in_base()
    hold_quat = ee_quat.clone()
    closed = torch.full((n,), GRIPPER_CLOSE, device=dev)
    phases = [(30, 0.015), (40, -0.004)]  # (steps, height above the rest surface)
    for steps, dz in phases:
        target_local = torch.cat([press_xy, (rest_top + dz)[:, None]], dim=-1)
        target_base = target_local - base_pos
        for _ in range(steps):
            ee_pos, ee_quat, *_ = env.poses_in_base()
            act = relative_action(ee_pos, ee_quat, target_base, hold_quat, closed,
                                  cfg.env.ik_pos_scale, cfg.env.ik_rot_scale)
            obs = step(act)
    ee_pos, *_ = env.poses_in_base()
    reach_err = torch.linalg.norm(ee_pos - (torch.cat([press_xy, (rest_top - 0.004)[:, None]], -1) - base_pos), dim=-1)
    if not soft:
        # The tool is driven 4 mm "into" the rigid pad: it must stop on the surface, and nothing may glitch.
        tip_h = ee_pos[:, 2] + base_pos[2] - rest_top
        check("tissue/solid_when_pressed", bool((tip_h > -2e-3).all()),
              f"tool tip height above the pad while pressing {f(tip_h * 1000)} mm (want > -2: it must not sink in)")
        glitch = terminated_any & ~ended_before_press
        check("tissue/no_glitch_when_pressed", not bool(glitch.any()),
              "no episode ended while the tool pressed on the pad (a physics glitch would end it)")
    press_nodes = tissue.data.nodal_pos_w.torch if soft else None
    rest_local = rest - origins[:, None, :] if soft else None
    if soft:
        near = (torch.linalg.norm(rest_local[..., :2] - press_xy[:, None, :], dim=-1) < 0.012) & \
               (rest_z >= rest_z.max(dim=1, keepdim=True).values - 1e-4)
        dent = ((rest_z - press_nodes[..., 2]) * near).max(dim=1).values
        check("tissue/dents_when_pressed", bool((dent > 1e-3).all()),
              f"max dent under the tool {f(dent * 1000)} mm (want > 1), tool tip {f(reach_err * 1000)} mm from its target",
              dent_m=f(dent), reach_error_m=f(reach_err))
        slide = torch.linalg.norm((press_nodes - rest)[..., :2], dim=-1).mean(dim=1)
        check("tissue/no_sliding", bool((slide < 2e-3).all()), f"mean sideways node displacement {f(slide * 1000)} mm (want < 2)")

    # ------------------------------------------------------------------ outputs
    rows = []
    obs_np = snapshot.cpu()
    for i in range(n):
        left, right = vis.to_hwc(obs_np[i, 0]), vis.to_hwc(obs_np[i, 1])
        l_mark, r_mark, ana = left.copy(), right.copy(), vis.anaglyph(left, right)
        for j, color in ((0, (255, 255, 0)), (1, (0, 255, 255))):
            vis.draw_cross(l_mark, float(uv_l[i, j, 0]), float(uv_l[i, j, 1]), color, size=2)
            vis.draw_cross(r_mark, float(uv_r[i, j, 0]), float(uv_r[i, j, 1]), color, size=2)
        dep = vis.depth_to_rgb(snapshot_depth[i, 0].cpu().numpy())
        rows.append(np.concatenate([l_mark, r_mark, ana, dep], axis=1))
    vis.save_png(vis.upscale(np.concatenate(rows, axis=0), 4), out / "stereo_check.png")
    vis.save_mp4(video, out / "stereo_check.mp4", fps=fps)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    n_fail = sum(not r["passed"] for r in results)
    print("\n" + "=" * 72)
    print(f"{len(results) - n_fail}/{len(results)} checks passed. Outputs in {out.resolve()}")
    for r in results:
        if not r["passed"]:
            print(f"  FAILED: {r['check']} -- {r['detail']}")
    print("=" * 72, flush=True)
    env.close()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
