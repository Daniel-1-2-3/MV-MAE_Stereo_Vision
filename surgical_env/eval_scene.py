"""Evaluation-time scene tools: env_0's paper camera and the robustness suite's scene changes.

Import only after the simulator has been launched (see sim_app.py).

* ``spawn_view_camera`` + ``HiResRecorder``: a render-only camera prim in env_0 (created after the scene was cloned,
  so no other env has it). Its render product - and one on the left stereo camera at a higher resolution - exists
  only between ``start()`` and ``stop()`` of an evaluation episode, so it is never rendered while training and the
  policy's tiled stereo cameras are untouched.
* Scene perturbations: the whole stereo rig orbited about its look-at point, the dome light's intensity scaled, a
  shadow-casting distant light switched on. Each saves the exact USD values it changes (the same xformOp attributes
  Isaac Sim's XFormPrim.set_local_poses writes) and returns a function that writes them back and checks them, or
  raises RuntimeError with the reason when it cannot do so.

Isaac Lab 3.0: its spawners take orientations as (x, y, z, w) (converted from this project's (w, x, y, z) with
``xyzw_from_wxyz``); USD's own Gf quaternions are unaffected (real part first, as before).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import vis

from .rotations import xyzw_from_wxyz
from .stereo import make_stereo_rig, quat_ros_to_opengl


def _stage():
    import isaaclab.sim as sim_utils

    return sim_utils.get_current_stage()


def spawn_view_camera(prim_path: str, pos_local, quat_ros, width: int, height: int, focal_length: float,
                      horizontal_aperture: float, clipping=(0.005, 5.0)) -> None:
    """USD pinhole camera at `prim_path` (pose relative to its parent, ROS convention). No render product."""
    import isaaclab.sim as sim_utils

    cfg = sim_utils.PinholeCameraCfg(
        focal_length=focal_length, focus_distance=400.0, horizontal_aperture=horizontal_aperture,
        vertical_aperture=horizontal_aperture * height / width, clipping_range=tuple(clipping), lock_camera=False)
    cfg.func(prim_path, cfg, translation=tuple(float(v) for v in pos_local),
             orientation=tuple(float(v) for v in xyzw_from_wxyz(quat_ros_to_opengl(quat_ros))))


class HiResRecorder:
    """Records env_0's paper cameras into H.264 mp4 files during one evaluation episode.

    cams: (name, camera prim path, (width, height), mp4 path). start() creates one render product + RGB annotator per
    camera, capture() appends the latest image of each to its video, stop() destroys the render products (so they are
    not rendered while training) and returns {name: mp4 path} for the videos that got frames. capture() never raises:
    a problem is printed once and recording stops.
    """

    def __init__(self, cams, fps: float, crf: int, preset: str, check=None):
        self.cams, self.fps, self.crf, self.preset = cams, fps, crf, preset
        self.check = check  # optional: () -> stereo images, to verify the policy's images do not change
        self.live: list = []
        self.error: str | None = None
        self.destroy_failed = False

    def start(self) -> None:
        import omni.replicator.core as rep

        before = None
        if self.check is not None:
            try:
                before = self.check()
            except Exception as e:
                print(f"hi-res camera check skipped: {type(e).__name__}: {e}")
        for name, prim, res, path in self.cams:
            rp = rep.create.render_product(prim, resolution=tuple(int(v) for v in res))
            rp_path = rp if isinstance(rp, str) else rp.path
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([rp_path])
            self.live.append({"name": name, "rp": rp, "rp_path": rp_path, "annot": annot, "res": tuple(res),
                              "writer": vis.Mp4Writer(path, self.fps, self.crf, self.preset)})
        if before is not None:
            try:
                change = float((self.check().float() - before.float()).abs().max())
                print(f"hi-res camera check: largest change of the policy's stereo pixels when the hi-res render "
                      f"products were added: {change:.1f} grey levels" + (" (OK)" if change < 1.0 else
                                                                          " (WARNING: changed)"), flush=True)
            except Exception as e:
                print(f"hi-res camera check skipped: {type(e).__name__}: {e}")

    def capture(self) -> None:
        if self.error is not None:
            return
        try:
            for cam in self.live:
                data = cam["annot"].get_data()
                if isinstance(data, dict):
                    data = data.get("data")
                img = np.asarray(data)
                w, h = cam["res"]
                if img.ndim != 3 or img.shape[0] != h or img.shape[1] != w:
                    continue  # nothing rendered into it yet
                cam["writer"].append(img[..., :3].astype(np.uint8))
        except Exception as e:  # a recording problem must never stop the evaluation
            self.error = f"{type(e).__name__}: {e}"
            print(f"hi-res capture failed, recording stopped: {self.error}", flush=True)

    def stop(self) -> dict[str, Path]:
        out = {}
        for cam in self.live:
            try:
                cam["annot"].detach([cam["rp_path"]])
            except Exception as e:
                print(f"hi-res annotator detach failed: {e}")
            try:
                cam["rp"].destroy()
            except Exception as e:  # it would keep rendering during training
                self.destroy_failed = True
                print(f"hi-res render product could not be destroyed ({e})")
            cam["writer"].close()
            if cam["writer"].frames:
                out[cam["name"]] = cam["writer"].path
        self.live = []
        return out


# ======================================================================== scene perturbations
class _Xform:
    """A prim's xformOp:translate / xformOp:orient: saved exactly, set, written back."""

    def __init__(self, prim):
        from pxr import UsdGeom

        if not prim or not prim.IsValid():
            raise RuntimeError("camera prim not found")
        ops = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
        if not {"xformOp:translate", "xformOp:orient"} <= set(ops) <= {"xformOp:translate", "xformOp:orient",
                                                                        "xformOp:scale"}:
            raise RuntimeError(f"{prim.GetPath()} has transform ops {ops}, expected translate / orient (/ scale)")
        self.path = str(prim.GetPath())
        self.t, self.o = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient")
        self.t0, self.o0 = self.t.Get(), self.o.Get()

    def position(self) -> np.ndarray:
        return np.array([self.t0[0], self.t0[1], self.t0[2]], dtype=np.float64)

    def quat(self) -> np.ndarray:
        return np.array([self.o0.GetReal(), *self.o0.GetImaginary()], dtype=np.float64)

    def set(self, pos, quat_wxyz) -> None:
        self.t.Set(type(self.t0)(*(float(v) for v in pos)))
        self.o.Set(type(self.o0)(*(float(v) for v in quat_wxyz)))

    def restore(self) -> None:
        self.t.Set(self.t0)
        self.o.Set(self.o0)
        if self.t.Get() != self.t0 or self.o.Get() != self.o0:
            raise RuntimeError(f"could not restore the pose of {self.path}")


def orbit_stereo_rig(env_prim_paths, rig, cam_cfg, d_azimuth_deg: float, d_elevation_deg: float):
    """Move both stereo cameras of every env as one rig: azimuth / elevation about the look-at point changed by the
    given degrees, still aimed at it. Returns the restore function."""
    new = make_stereo_rig(dataclasses.replace(cam_cfg, azimuth_deg=cam_cfg.azimuth_deg + d_azimuth_deg,
                                              elevation_deg=cam_cfg.elevation_deg + d_elevation_deg), rig.look_at)
    q_nominal, q_new = quat_ros_to_opengl(rig.quat_ros), quat_ros_to_opengl(new.quat_ros)
    stage = _stage()
    moves = []
    for env_path in env_prim_paths:
        for name, nominal_pos, new_pos in (("StereoLeft", rig.left_pos, new.left_pos),
                                           ("StereoRight", rig.right_pos, new.right_pos)):
            x = _Xform(stage.GetPrimAtPath(f"{env_path}/{name}"))
            # self-check: the prims must be where the design puts them, else the computed move would be wrong
            if np.abs(x.position() - np.asarray(nominal_pos)).max() > 1e-5 or abs(float(x.quat() @ q_nominal)) < 1 - 1e-5:
                raise RuntimeError(f"{x.path} is not at its designed pose (moved by something else?)")
            moves.append((x, new_pos))
    done = []
    try:
        for x, pos in moves:
            x.set(pos, q_new)
            done.append(x)
    except Exception:
        for x in done:
            x.restore()
        raise

    def restore():
        for x in done:
            x.restore()

    return restore


def scale_dome_light(factor: float, prim_path: str = "/World/light"):
    attr = _stage().GetPrimAtPath(prim_path).GetAttribute("inputs:intensity")
    if not attr or attr.Get() is None:
        raise RuntimeError(f"no dome light intensity at {prim_path}")
    old = attr.Get()
    attr.Set(type(old)(old * factor))

    def restore():
        attr.Set(old)
        if attr.Get() != old:
            raise RuntimeError("could not restore the dome light intensity")

    return restore


def add_shadow_light(env_cfg, prim_path: str = "/World/robust_shadow_light"):
    """The shadow-casting distant light of env.shadow_light=true, added for one condition and removed after it."""
    if env_cfg.shadow_light:
        raise RuntimeError("the shadow light is already on in this run (it is part of the nominal scene)")
    import isaaclab.sim as sim_utils

    from .env_cfg import _down_light_quat

    stage = _stage()
    cfg = sim_utils.DistantLightCfg(intensity=env_cfg.shadow_light_intensity, angle=0.5)
    cfg.func(prim_path, cfg,
             orientation=xyzw_from_wxyz(_down_light_quat(env_cfg.shadow_light_tilt_deg, env_cfg.shadow_light_azimuth_deg)))

    def restore():
        stage.RemovePrim(prim_path)
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise RuntimeError("could not remove the added shadow light")

    return restore
