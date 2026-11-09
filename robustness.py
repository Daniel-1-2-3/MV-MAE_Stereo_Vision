"""Robustness test suite: the current policy under perturbed images and perturbed scenes.

No simulator imports. Image and frame-stack perturbations are plain torch on the observation the policy sees (the
robot state, if used, is never touched). Scene perturbations (stereo rig pose, lighting) are asked of the environment
(``StereoNeedleEnv.sim_perturbation``); an environment that cannot apply one and restore it exactly says why, and
that condition is skipped. The suite therefore also runs on the CPU with the fake environment of the tests.

It never touches training:
  * the policy acts deterministically (actor mean) under torch.no_grad; nothing goes into the replay memory and no
    parameter is updated;
  * perturbation noise comes from torch.Generator objects of its own; the global CPU / CUDA / numpy / python random
    states are saved before the suite and restored after it (in between they are re-seeded so that every condition
    starts from the same needle placements);
  * the environment's step counters and reward weights (Isaac Lab's reward curriculum reads the step counter) are
    restored, and every scene change is undone before the next condition starts.

Each condition runs `episodes` deterministic episodes: rounds of one episode in every parallel env, round r reset
with seed robust.seed + r for every condition, so all conditions (and the suite's own "nominal" run) see the same
needle placements.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agent.replay import FrameStacker

METRICS = ("success", "success_any", "return", "final_goal_distance")
SIM_CHANGE_MIN = 0.5  # mean |difference| (grey levels) of the first frame below which a scene change "did not render"


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    severity: float  # x of the family's plot; 0 = no perturbation (identity)
    kind: str  # "image" (per frame), "temporal" (frame stack), "sim" (scene), "none" (nominal)
    variant: str = ""


# family -> (kind, x-axis label of its success-vs-severity plot; "" = categorical, bar chart only)
FAMILIES: dict[str, tuple[str, str]] = {
    "view_drop": ("image", ""),
    "view_duplicate": ("image", ""),
    "gaussian_noise": ("image", "noise std (fraction of 255)"),
    "occlusion_one": ("image", "occluded fraction of one view"),
    "occlusion_both": ("image", "occluded fraction of each view"),
    # MV-MAE only, "reconstruct, then act": the same removed pixels as the base family (view_drop, occlusion_*), filled
    # in by the MV-MAE reconstruction before the policy sees the observation (see mvmae_fill)
    "view_drop_recon": ("image", ""),
    "occlusion_one_recon": ("image", "occluded fraction of one view (MV-MAE fills it in)"),
    "occlusion_both_recon": ("image", "occluded fraction of each view (MV-MAE fills it in)"),
    "brightness": ("image", "brightness factor - 1"),
    "contrast": ("image", "contrast factor - 1"),
    "hue": ("image", "hue shift (fraction of the colour wheel)"),
    "blur": ("image", "Gaussian blur sigma (px)"),
    "temporal": ("temporal", ""),
    "camera_azimuth": ("sim", "stereo rig azimuth offset (deg)"),
    "camera_elevation": ("sim", "stereo rig elevation offset (deg)"),
    "light_intensity": ("sim", "dome light factor - 1"),
    "shadow_light": ("sim", ""),
}


def _c(name: str, family: str, severity: float, variant: str = "") -> Condition:
    return Condition(name, family, float(severity), FAMILIES[family][0], variant)


NOMINAL = Condition("nominal", "nominal", 0.0, "none")
CONDITIONS: tuple[Condition, ...] = (
    _c("view_drop_left", "view_drop", 1.0, "left"),  # left view black
    _c("view_drop_right", "view_drop", 1.0, "right"),
    _c("view_duplicate_left", "view_duplicate", 1.0, "left_to_right"),  # right view replaced by the left: no stereo depth
    _c("view_swap", "view_duplicate", 1.0, "swap"),  # left and right exchanged
    *(_c(f"gaussian_noise_{s:.2f}", "gaussian_noise", s) for s in (0.05, 0.10, 0.20)),
    *(_c(f"occlusion_one_{p}", "occlusion_one", p / 100) for p in (10, 25, 40)),  # grey square in one random view
    *(_c(f"occlusion_both_{p}", "occlusion_both", p / 100) for p in (10, 25, 40)),  # in both, independent positions
    _c("view_drop_recon_left", "view_drop_recon", 1.0, "left"),  # view_drop_left, MV-MAE redraws the left view
    _c("view_drop_recon_right", "view_drop_recon", 1.0, "right"),
    *(_c(f"occlusion_one_recon_{p}", "occlusion_one_recon", p / 100) for p in (10, 25, 40)),
    *(_c(f"occlusion_both_recon_{p}", "occlusion_both_recon", p / 100) for p in (10, 25, 40)),
    _c("brightness_0.6", "brightness", -0.4),
    _c("brightness_1.4", "brightness", 0.4),
    _c("contrast_0.6", "contrast", -0.4),
    _c("hue_shift_0.10", "hue", 0.10),
    _c("blur_1", "blur", 1.0),
    _c("blur_2", "blur", 2.0),
    _c("frame_drop", "temporal", 1.0, "frame_drop"),  # newest frame repeated in the stack (one dropped frame)
    _c("frozen_frames_2", "temporal", 1.0, "frozen"),  # images refreshed only every 2nd step
    *(_c(f"camera_azimuth_{d:+d}", "camera_azimuth", d) for d in (-10, -5, 5, 10)),
    *(_c(f"camera_elevation_{d:+d}", "camera_elevation", d) for d in (-5, 5)),
    _c("light_dome_0.5", "light_intensity", -0.5),
    _c("light_dome_1.5", "light_intensity", 0.5),
    _c("light_shadow_on", "shadow_light", 1.0),
)
BY_NAME = {c.name: c for c in CONDITIONS}
# MV-MAE only; left out of mean_success / mean_success_rel, so the headline mean covers the same conditions for every agent
RECON_FAMILIES = ("view_drop_recon", "occlusion_one_recon", "occlusion_both_recon")


def resolve_conditions(names) -> list[Condition]:
    """Condition / family names -> conditions in registry order (scene changes last). Unknown names raise."""
    wanted = set()
    for n in names:
        if n in BY_NAME:
            wanted.add(n)
        elif n in FAMILIES:
            wanted |= {c.name for c in CONDITIONS if c.family == n}
        else:
            raise ValueError(f"unknown robustness condition or family '{n}'. Families: {', '.join(FAMILIES)}; "
                             f"conditions: {', '.join(BY_NAME)}")
    picked = [c for c in CONDITIONS if c.name in wanted]
    return [c for c in picked if c.kind != "sim"] + [c for c in picked if c.kind == "sim"]


# ======================================================================== image perturbations (plain torch)
def episode_state(cond: Condition, num_envs: int, obs_shape, device, gen: torch.Generator | None) -> dict:
    """Random choices that stay fixed for a whole episode (occluder position and, for occlusion_one, the view)."""
    family = cond.family.removesuffix("_recon")  # a _recon condition removes exactly its base condition's pixels
    if family not in ("occlusion_one", "occlusion_both") or cond.severity <= 0:
        return {}
    v, _, h, w = obs_shape
    side = min(h, w, int(round(math.sqrt(cond.severity * h * w))))  # square covering `severity` of the image
    if side <= 0:
        return {}
    y0 = torch.randint(0, h - side + 1, (num_envs, v), generator=gen, device=device)
    x0 = torch.randint(0, w - side + 1, (num_envs, v), generator=gen, device=device)
    rows = torch.arange(h, device=device)[None, None, :, None]
    cols = torch.arange(w, device=device)[None, None, None, :]
    mask = ((rows >= y0[..., None, None]) & (rows < y0[..., None, None] + side)
            & (cols >= x0[..., None, None]) & (cols < x0[..., None, None] + side))  # (N, V, H, W)
    if family == "occlusion_one":
        view = torch.randint(0, v, (num_envs,), generator=gen, device=device)
        mask &= (torch.arange(v, device=device)[None] == view[:, None])[..., None, None]
    return {"occluder": mask[:, :, None]}  # (N, V, 1, H, W)


def _hue_matrix(turns: float, device) -> torch.Tensor:
    """Rotation of RGB colours about the grey axis by `turns` of the colour wheel (keeps greys grey)."""
    a = 2.0 * math.pi * turns
    k = torch.full((3,), 1.0 / math.sqrt(3.0), dtype=torch.float64)
    kx = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=torch.float64)
    r = math.cos(a) * torch.eye(3, dtype=torch.float64) + math.sin(a) * kx + (1 - math.cos(a)) * torch.outer(k, k)
    return r.float().to(device)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(N, V, 3, H, W) float -> same, separable Gaussian blur with reflected borders."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    n, v, c, h, w = x.shape
    y = x.reshape(n * v * c, 1, h, w)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return y.reshape(n, v, c, h, w)


def perturb_images(obs: torch.Tensor, cond: Condition, state: dict | None = None,
                   gen: torch.Generator | None = None) -> torch.Tensor:
    """One stereo frame per env, (N, V, 3, H, W) uint8 -> perturbed uint8 copy (the input itself at severity 0).

    Severity meaning per family: view_drop - fraction of the view's intensity removed (1 = black); view_duplicate -
    blend of the replaced view(s) towards the source (1 = fully replaced / swapped); gaussian_noise - std as a fraction
    of 255; occlusion_* - area fraction of the grey (128) square (position from `state`, see episode_state);
    brightness / contrast - factor - 1; hue - turns of the colour wheel; blur - sigma in pixels.
    """
    s, fam = cond.severity, cond.family.removesuffix("_recon")  # a _recon condition: its base condition's pixels
    if cond.kind != "image" or s == 0:
        return obs
    x = obs.float()
    if fam == "view_drop":
        x[:, 0 if cond.variant == "left" else 1] *= 1.0 - s
    elif fam == "view_duplicate":
        left, right = x[:, 0].clone(), x[:, 1].clone()
        if cond.variant == "swap":
            x[:, 0] = (1 - s) * left + s * right
        x[:, 1] = (1 - s) * right + s * left
    elif fam == "gaussian_noise":
        x = x + torch.randn(x.shape, generator=gen, device=x.device) * (s * 255.0)
    elif fam in ("occlusion_one", "occlusion_both"):
        state = state if state is not None else episode_state(cond, obs.shape[0], obs.shape[1:], obs.device, gen)
        if "occluder" in state:
            x = torch.where(state["occluder"], torch.full_like(x, 128.0), x)
    elif fam == "brightness":
        x = x * (1.0 + s)
    elif fam == "contrast":
        grey = (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]).mean(dim=(-2, -1))  # (N, V)
        m = grey[..., None, None, None]
        x = m + (x - m) * (1.0 + s)
    elif fam == "hue":
        x = torch.einsum("ij,nvjhw->nvihw", _hue_matrix(s, x.device), x)
    elif fam == "blur":
        x = _gaussian_blur(x, s)
    else:
        raise ValueError(f"not an image perturbation: {cond.name}")
    return x.round().clamp(0, 255).to(torch.uint8)


# ------------------------------------------------------------------ "reconstruct, then act" (MV-MAE only)
def missing_pixels(cond: Condition, state: dict, num_envs: int, obs_shape, device) -> torch.Tensor:
    """(N, V, H, W) bool: the pixels `cond` removes (view_drop*: the whole dropped view; occlusion*: the occluder)."""
    v, _, h, w = obs_shape
    missing = torch.zeros(num_envs, v, h, w, dtype=torch.bool, device=device)
    family = cond.family.removesuffix("_recon")
    if family == "view_drop":
        missing[:, 0 if cond.variant == "left" else 1] = True
    elif family in ("occlusion_one", "occlusion_both") and "occluder" in state:
        missing = state["occluder"][:, :, 0].to(device)
    return missing


def hidden_tokens(m, missing: torch.Tensor) -> torch.Tensor:
    """(N, V, H, W) missing pixels -> (N, L) bool hidden MV-MAE tokens, order (frame, view, row, col) as MVMAE._tokens.

    A token is hidden if any pixel of its p x p patch is missing; the pixel mask is the same in every frame.
    """
    n = missing.shape[0]
    per_view = missing.reshape(n, m.V, m.gh, m.p, m.gw, m.p).any(dim=5).any(dim=3)  # (N, V, gh, gw)
    return per_view.reshape(n, 1, m.V * m.N).expand(n, m.F, m.V * m.N).reshape(n, m.L)


@torch.no_grad()
def mvmae_fill(agent, stack: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
    """(N, F, V, 3, H, W) uint8 frame stack -> copy whose `missing` pixels ((N, V, H, W), the same in every frame) are
    replaced by the MV-MAE reconstruction; every real pixel is kept.

    MVMAE.forward_mae with this deterministic mask instead of view_mask(): only the tokens with no missing pixel go
    through the encoder, the decoder redraws the rest. The stack is never modified (FrameStacker returns its buffer).
    """
    m = agent.mvmae
    dev = next(m.parameters()).device
    x, miss = stack.to(dev), missing.to(dev)
    out = x.clone()
    hid = hidden_tokens(m, miss)
    counts = (~hid).sum(dim=1)  # visible tokens per env (differs with the occluder's alignment to the patch grid)
    with agent._autocast():
        tokens = m._tokens(m.normalize(x))
        for k in counts.unique().tolist():
            if k == m.L:
                continue  # nothing missing
            idx = (counts == k).nonzero().squeeze(1)
            b, dim = idx.numel(), tokens.shape[-1]
            keep_idx = torch.argsort(hid[idx].to(torch.int8), dim=1, stable=True)[:, :k]  # visible positions, ascending
            if k > 0:
                visible = torch.gather(tokens[idx], 1, keep_idx[..., None].expand(-1, -1, dim))
                dec_visible = m.dec_in(m._run_encoder(visible))
                d = dec_visible.shape[-1]
                full = m.mask_token.to(dec_visible.dtype).expand(b, m.L, d).clone()
                full.scatter_(1, keep_idx[..., None].expand(-1, -1, d), dec_visible)
            else:  # every token missing: the decoder starts from mask tokens alone
                d = m.mask_token.shape[-1]
                full = m.mask_token.to(tokens.dtype).expand(b, m.L, d).clone()
            full = full.reshape(b, m.F, m.V, m.N, d)
            full = (
                full
                + m.dec_pos.to(full.dtype)
                + m.dec_view.to(full.dtype)[None, None, :, None, :]
                + m.dec_frame.to(full.dtype)[None, :, None, None, :]
            ).reshape(b, m.L, d)
            for blk in m.decoder:
                full = blk(full)
            full = m.dec_norm(full)
            recon = m.denormalize(m.unpatchify(m.pixel_head(full).float()))  # (b, F, V, 3, H, W) uint8
            out[idx] = torch.where(miss[idx][:, None, :, None], recon, x[idx])
    return out.to(stack.device)


def drop_frames(stack: torch.Tensor, dropped: int) -> torch.Tensor:
    """(N, F, ...) frame stack -> copy whose `dropped` frames before the newest are replaced by the newest
    (a camera that delivered the same image again). 0 returns the stack itself."""
    dropped = min(int(dropped), stack.shape[1] - 1)
    if dropped <= 0:
        return stack
    out = stack.clone()
    out[:, -1 - dropped:-1] = stack[:, -1:]
    return out


class PerturbedObservations:
    """What the policy sees under a condition: perturbed frames go into a normal frame stack.

    temporal/frozen: the camera image is refreshed only every (severity + 1)-th step (the frame stack keeps
    receiving the last refreshed image); temporal/frame_drop: in every stack the policy gets, `severity` frames
    before the newest are replaced by the newest. `fed` is the frame that went into the stack this step.
    With `fill` (fill(stack, missing) -> stack, e.g. mvmae_fill), the removed pixels are filled in every step from the
    perturbed frames the stack holds; `shown` is the newest filled frame (what the policy saw).
    """

    def __init__(self, cond: Condition, num_envs: int, frame_stack: int, obs_shape, device, gen, fill=None):
        self.cond, self.gen, self.fill = cond, gen, fill
        self.stacker = FrameStacker(num_envs, frame_stack, obs_shape, device)
        self.state = episode_state(cond, num_envs, obs_shape, device, gen)
        self.missing = missing_pixels(cond, self.state, num_envs, obs_shape, device) if fill is not None else None
        self.t = 0
        self.fed = None
        self.shown = None

    def _frame(self, obs: torch.Tensor) -> torch.Tensor:
        return perturb_images(obs, self.cond, self.state, self.gen)

    def _view(self, stack: torch.Tensor) -> torch.Tensor:
        if self.cond.variant == "frame_drop":
            return drop_frames(stack, int(self.cond.severity))
        if self.fill is not None:
            stack = self.fill(stack, self.missing)
            self.shown = stack[:, -1]
        return stack

    def reset(self, obs: torch.Tensor) -> torch.Tensor:
        self.t = 0
        self.fed = self._frame(obs)
        return self._view(self.stacker.reset(self.fed))

    def step(self, obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        self.t += 1
        frame = self._frame(obs)
        period = int(self.cond.severity) + 1 if self.cond.variant == "frozen" else 1
        if period > 1 and self.t % period != 0:
            frame = torch.where(done[:, None, None, None, None], frame, self.fed)  # a new episode gets a fresh image
        self.fed = frame
        return self._view(self.stacker.step(frame, done))


# ======================================================================== running the suite
@contextlib.contextmanager
def preserved_random_state():
    """Save the global CPU / CUDA / numpy / python random states and put them back afterwards."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(np_state)
        random.setstate(py_state)


def condition_generator(cond: Condition, seed: int, rnd: int, device) -> torch.Generator:
    # a _recon condition draws from its base condition's generator, so its occluders land in exactly the same places
    name = cond.name.replace("_recon", "", 1) if cond.family in RECON_FAMILIES else cond.name
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) * 1_000_003 + zlib.crc32(f"{name}/{rnd}".encode())) % (2 ** 62))
    return g


def _stack_rows(frames, scale: int) -> np.ndarray:
    import vis

    return np.concatenate([vis.stereo_frame(f, scale) for f in frames], axis=0)


@torch.no_grad()
def run_episodes(env, agent, env_step: int, cond: Condition, gen, record: bool, video_scale: int = 2) -> dict:
    """One deterministic episode in every parallel env under `cond` (scene changes must already be applied).

    Returns per-env tensors (CPU) for METRICS, `first_obs` (the unperturbed first image, on the env device) and, if
    `record`, `frames`: the stereo input the policy received in one episode (a success if there is one, else one
    that reached success at some point, else env 0); temporal conditions show the whole frame stack (oldest on top),
    _recon conditions the filled-in frame the policy saw.
    """
    from trainer import robot_state

    agent.train(False)
    n, dev = env.num_envs, env.device
    obs = env.reset()
    first_obs = obs.clone()
    fill = (lambda s, m: mvmae_fill(agent, s, m)) if cond.family in RECON_FAMILIES else None
    pipe = PerturbedObservations(cond, n, agent.frame_stack, env.obs_shape, dev, gen, fill=fill)
    stack = pipe.reset(obs)
    seen = (lambda: pipe.shown) if fill is not None else (lambda: pipe.fed)  # what the policy saw, for the video
    fed = [seen().clone()] if record else []
    valid = torch.ones(n, dtype=torch.long, device=dev)
    ret = torch.zeros(n, device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    success_any = torch.zeros(n, dtype=torch.bool, device=dev)
    goal_dist = torch.zeros(n, device=dev)
    prop = robot_state(env, agent)  # the robot state is never perturbed
    for _ in range(env.max_episode_steps + 1):
        action = agent.act(stack, env_step, eval_mode=True, proprio=prop)
        obs, reward, term, trunc, info = env.step(action)
        done = term | trunc
        alive = ~finished
        live = alive & ~done
        ret += reward * alive
        success_any |= info["success_now"] & live
        newly = done & alive
        success = torch.where(newly, info["final_success"], success)
        success_any |= newly & info["final_success"]
        goal_dist = torch.where(newly, info["final_goal_distance"], goal_dist)
        finished |= done
        stack = pipe.step(obs, done)
        if record:
            fed.append(seen().clone())
            valid += live.long()
        prop = robot_state(env, agent)
        if bool(finished.all()):
            break
    out = {"success": success.cpu(), "success_any": success_any.cpu(), "return": ret.cpu(),
           "final_goal_distance": goal_dist.cpu(), "first_obs": first_obs}
    if record:
        if bool(success.any()):
            pick, ok = int(success.nonzero()[0]), 1.0
        elif bool(success_any.any()):
            pick, ok = int(success_any.nonzero()[0]), 0.5
        else:
            pick, ok = 0, 0.0
        clip = torch.stack([f[pick] for f in fed[: int(valid[pick])]])  # (T, V, 3, H, W) frames fed to the stack
        if cond.kind == "temporal":
            stacker = FrameStacker(1, agent.frame_stack, env.obs_shape, clip.device)
            stacks = [pipe._view(stacker.reset(clip[:1]))[0]]
            stacks += [pipe._view(stacker.step(clip[t:t + 1], torch.zeros(1, dtype=torch.bool, device=clip.device)))[0]
                       for t in range(1, len(clip))]
            out["frames"] = [_stack_rows(s, video_scale) for s in stacks]
        else:
            import vis

            out["frames"] = [vis.stereo_frame(f, video_scale) for f in clip]
        out["video_env"], out["video_success"] = pick, ok
    agent.train(True)
    return out


def _mean(x: torch.Tensor) -> float:
    return float(x.float().mean())


def _ratio(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def run_suite(env, agent, env_step: int, rcfg, conditions: list[Condition], video_conditions=(), episodes: int = 0,
              video_dir: str | Path | None = None, video_scale: int = 2, log=print) -> dict:
    """Run the nominal setting plus every condition on identical needle placements; returns every number.

    `rcfg` is a RobustConfig (seed). Videos for the conditions in `video_conditions` go to
    <video_dir>/robust_<condition>_<env_step>.mp4.
    """
    import vis

    t_start = time.time()
    n = env.num_envs
    episodes = int(episodes) or n
    rounds = math.ceil(episodes / n)
    video_conditions = set(video_conditions)
    results = {"env_step": int(env_step), "seed": int(rcfg.seed), "episodes_per_condition": episodes, "num_envs": n,
               "rounds": rounds, "conditions": {}}
    saved_env = env.training_state() if hasattr(env, "training_state") else None
    nominal_first = None
    try:
        with preserved_random_state():
            for cond in [NOMINAL] + list(conditions):
                entry = {"family": cond.family, "severity": cond.severity, "kind": cond.kind, "variant": cond.variant,
                         "skipped": None}
                results["conditions"][cond.name] = entry
                restore = None
                if cond.family in RECON_FAMILIES and not getattr(agent, "uses_mvmae", False):
                    entry["skipped"] = (f"needs the MV-MAE reconstruction (agent.encoder="
                                        f"{getattr(agent, 'encoder_kind', '?')})")
                    log(f"  robust: skipping {cond.name} ({entry['skipped']})")
                    continue
                if cond.kind == "sim":
                    if not hasattr(env, "sim_perturbation"):
                        entry["skipped"] = "this environment has no scene perturbations"
                    else:
                        try:
                            restore = env.sim_perturbation(cond.family, cond.severity)
                        except Exception as e:  # cannot be applied and restored exactly: skip, say why
                            entry["skipped"] = f"{type(e).__name__}: {e}"
                    if entry["skipped"]:
                        log(f"  robust: skipping {cond.name} ({entry['skipped']})")
                        continue
                try:
                    per = {k: [] for k in METRICS}
                    for r in range(rounds):
                        torch.manual_seed(int(rcfg.seed) + r)  # same needle placements for every condition
                        np.random.seed((int(rcfg.seed) + r) % (2 ** 32))
                        random.seed(int(rcfg.seed) + r)
                        record = r == 0 and cond.name in video_conditions and video_dir is not None
                        gen = condition_generator(cond, rcfg.seed, r, env.device)
                        out = run_episodes(env, agent, env_step, cond, gen, record, video_scale)
                        for k in METRICS:
                            per[k].append(out[k])
                        if r == 0 and cond is NOMINAL:
                            nominal_first = out["first_obs"]
                        if r == 0 and cond.kind == "sim" and nominal_first is not None:
                            change = float((out["first_obs"].float() - nominal_first.float()).abs().mean())
                            entry["image_change"] = change
                            if change < SIM_CHANGE_MIN:
                                entry["skipped"] = (f"no visible change in the rendered images (mean |difference| "
                                                    f"{change:.3f} grey levels): the scene change did not reach the "
                                                    f"renderer")
                                break
                        if record and out.get("frames"):
                            path = Path(video_dir) / f"robust_{cond.name}_{env_step}.mp4"
                            vis.save_mp4(out["frames"], path, fps=1.0 / env.step_dt)
                            entry["video"], entry["video_success"] = str(path), out["video_success"]
                finally:
                    if restore is not None:
                        restore()  # an exception here propagates: training must never continue on a changed scene
                if entry["skipped"]:
                    log(f"  robust: skipping {cond.name} ({entry['skipped']})")
                    continue
                vals = {k: torch.cat(per[k])[:episodes] for k in METRICS}
                entry["episodes"] = {k: [float(x) for x in v.float()] for k, v in vals.items()}
                for k in METRICS:
                    entry[k] = _mean(vals[k])
            if nominal_first is not None and any(c.kind == "sim" for c in conditions):
                # all scene changes undone: the first image of round 0 must look like the nominal one again
                torch.manual_seed(int(rcfg.seed))
                again = env.reset()
                results["restore_check_image_change"] = float((again.float() - nominal_first.float()).abs().mean())
                if results["restore_check_image_change"] >= SIM_CHANGE_MIN:
                    log(f"  robust: WARNING the scene looks different after restoring it (mean |difference| "
                        f"{results['restore_check_image_change']:.3f} grey levels)")
    finally:
        if saved_env is not None:
            env.restore_training_state(saved_env)
    nom = results["conditions"]["nominal"]
    for entry in results["conditions"].values():
        if entry["skipped"] is None:
            for k in METRICS:
                entry[f"{k}_rel"] = _ratio(entry[k], nom[k])
    done = [e for name, e in results["conditions"].items()
            if name != "nominal" and e["skipped"] is None and e["family"] not in RECON_FAMILIES]  # same set for all agents
    results["mean_success"] = float(np.mean([e["success"] for e in done])) if done else float("nan")
    results["mean_success_rel"] = _ratio(results["mean_success"], nom["success"]) if done else float("nan")
    results["time_s"] = time.time() - t_start
    return results


# ======================================================================== outputs
def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None  # strict JSON has no NaN
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def save_json(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(results), indent=1))
    return path


def family_curves(results: dict) -> dict[str, tuple[list[float], list[float]]]:
    """family -> (severities, success) sorted by severity, nominal at 0, for families with a numeric severity."""
    conds = results["conditions"]
    nominal = conds["nominal"].get("success")
    curves = {}
    for fam, (_, axis) in FAMILIES.items():
        pts = [(e["severity"], e["success"]) for e in conds.values()
               if e["family"] == fam and e["skipped"] is None and e.get("success") is not None]
        if not axis or not pts or nominal is None:
            continue
        pts = sorted(pts + [(0.0, nominal)])
        curves[fam] = ([p[0] for p in pts], [p[1] for p in pts])
    return curves


def summary_lines(results: dict) -> list[str]:
    nom = results["conditions"]["nominal"]
    lines = [f"robustness at {results['env_step']} env steps ({results['episodes_per_condition']} episodes per "
             f"condition, {results['time_s']:.0f}s): nominal success {nom['success']:.2f}"]
    for name, e in results["conditions"].items():
        if name == "nominal":
            continue
        if e["skipped"]:
            lines.append(f"  {name:22s} skipped: {e['skipped']}")
        else:
            lines.append(f"  {name:22s} success {e['success']:.2f} (x{e['success_rel']:.2f} of nominal) | "
                         f"success_any {e['success_any']:.2f} | return {e['return']:7.1f} | "
                         f"goal dist {e['final_goal_distance'] * 1000:5.1f} mm")
    return lines


def wandb_payload(results: dict, prefix: str = "robust") -> dict:
    """Scalars robust/<condition>/<metric> (+ _rel = condition / nominal), one line plot per severity family
    (success vs severity, nominal at 0), a bar chart of all conditions and the condition videos."""
    import wandb

    import vis

    step = results["env_step"]
    out: dict = {f"{prefix}/mean_success": results["mean_success"], f"{prefix}/mean_success_rel": results["mean_success_rel"],
                 f"{prefix}/time_s": results["time_s"]}
    if "restore_check_image_change" in results:
        out[f"{prefix}/restore_check_image_change"] = results["restore_check_image_change"]
    for name, e in results["conditions"].items():
        if e["skipped"] is not None:
            continue
        for k in METRICS:
            out[f"{prefix}/{name}/{k}"] = e[k]
            out[f"{prefix}/{name}/{k}_rel"] = e[f"{k}_rel"]
        if e.get("video"):
            try:
                out[f"{prefix}_video/{name}"] = vis.wandb_video(e["video"])
                out[f"{prefix}/{name}/video_success"] = e["video_success"]
            except Exception as err:  # a failed video must never stop training
                print(f"robust video logging failed ({name}): {err}")
    for fam, (xs, ys) in family_curves(results).items():
        table = wandb.Table(data=[[x, y] for x, y in zip(xs, ys)], columns=["severity", "success"])
        out[f"{prefix}_plot/{fam}"] = wandb.plot.line(
            table, "severity", "success", title=f"{fam}: success vs {FAMILIES[fam][1]} (step {step}, nominal at 0)")
    rows = [[name, e["success"]] for name, e in results["conditions"].items() if e["skipped"] is None]
    out[f"{prefix}_plot/all_conditions"] = wandb.plot.bar(
        wandb.Table(data=rows, columns=["condition", "success"]), "condition", "success",
        title=f"success per condition (step {step})")
    return out


# --------------------------------------------------------------------- matplotlib figures (PNG + PDF)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1-2 of the reference palette (validated pair)


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def save_figures(results_by_label: dict[str, dict], out_dir: str | Path, stem: str, formats=("png", "pdf")) -> list[Path]:
    """Success-vs-severity curves (one panel per family) and a grouped bar chart of every condition.

    `results_by_label`: {"MV-MAE": results, ...} (one or two runs). Returns the written files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(results_by_label)
    written = []
    curves = {lab: family_curves(r) for lab, r in results_by_label.items()}
    fams = [f for f in FAMILIES if any(f in c for c in curves.values())]
    if fams:
        cols = min(4, len(fams))
        rows = math.ceil(len(fams) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
        for ax, fam in zip(axes.flat, fams):
            for i, lab in enumerate(labels):
                if fam in curves[lab]:
                    xs, ys = curves[lab][fam]
                    ax.plot(xs, ys, color=SERIES[i % 2], linewidth=1.5, marker="o", markersize=4, label=lab)
            ax.axvline(0.0, color=INK_2, linewidth=0.6, linestyle=":")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(fam.replace("_", " "), fontsize=9)
            ax.set_xlabel(FAMILIES[fam][1], fontsize=8)
            ax.set_ylabel("success", fontsize=8)
            style_axes(ax)
        for ax in list(axes.flat)[len(fams):]:
            ax.axis("off")
        if len(labels) > 1:
            axes.flat[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        for fmt in formats:
            p = out_dir / f"{stem}_curves.{fmt}"
            fig.savefig(p, dpi=200)
            written.append(p)
        plt.close(fig)
    names = [n for n in results_by_label[labels[0]]["conditions"]]
    for r in list(results_by_label.values())[1:]:
        names += [n for n in r["conditions"] if n not in names]
    # a condition skipped in every run would look like 0 % success: leave it out
    names = [n for n in names if any(n in r["conditions"] and r["conditions"][n].get("skipped") is None
                                     for r in results_by_label.values())]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.32 * len(names) * max(1, len(labels))), 3.2))
    width = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        conds = results_by_label[lab]["conditions"]
        ys = [conds[n]["success"] if n in conds and conds[n]["skipped"] is None else np.nan for n in names]
        xs = np.arange(len(names)) + (i - (len(labels) - 1) / 2) * width
        ax.bar(xs, ys, width=width * 0.92, color=SERIES[i % 2], label=lab)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("success", fontsize=8)
    style_axes(ax)
    if len(labels) > 1:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for fmt in formats:
        p = out_dir / f"{stem}_bars.{fmt}"
        fig.savefig(p, dpi=200)
        written.append(p)
    plt.close(fig)
    return written
