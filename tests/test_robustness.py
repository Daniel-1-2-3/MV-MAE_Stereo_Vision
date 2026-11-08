"""Robustness suite: every perturbation (shape, dtype, determinism, identity at severity 0) and the suite's isolation
from training (random streams, step counters and scene restored; same needle placements for every condition)."""

import dataclasses
import json
import random

import numpy as np
import pytest
import torch

import robustness as rb
from robustness import CONDITIONS, PerturbedObservations, drop_frames, perturb_images

IMAGE_CONDS = [c for c in CONDITIONS if c.kind == "image"]


def frames(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 2, 3, 32, 32), generator=g, dtype=torch.uint8)


def gen(seed=7):
    return torch.Generator().manual_seed(seed)


def test_registry_names_families_and_resolution():
    assert len({c.name for c in CONDITIONS}) == len(CONDITIONS)
    assert all(c.family in rb.FAMILIES and rb.FAMILIES[c.family][0] == c.kind for c in CONDITIONS)
    every = rb.resolve_conditions(list(rb.FAMILIES))
    assert {c.name for c in every} == {c.name for c in CONDITIONS}
    assert [c.kind for c in every].index("sim") == len([c for c in CONDITIONS if c.kind != "sim"])  # scene changes last
    assert [c.name for c in rb.resolve_conditions(["blur", "gaussian_noise_0.10"])] == ["gaussian_noise_0.10", "blur_1",
                                                                                       "blur_2"]
    with pytest.raises(ValueError):
        rb.resolve_conditions(["not_a_condition"])
    from config import RobustConfig

    rb.resolve_conditions(RobustConfig().conditions)  # the config defaults are valid names
    rb.resolve_conditions(RobustConfig().video_conditions)
    # the spec'd severities
    names = {c.name for c in CONDITIONS}
    for n in ("view_drop_left", "view_drop_right", "view_duplicate_left", "view_swap", "gaussian_noise_0.05",
              "gaussian_noise_0.10", "gaussian_noise_0.20", "occlusion_one_10", "occlusion_one_25", "occlusion_one_40",
              "occlusion_both_10", "occlusion_both_25", "occlusion_both_40", "brightness_0.6", "brightness_1.4",
              "contrast_0.6", "hue_shift_0.10", "blur_1", "blur_2", "frame_drop", "frozen_frames_2",
              "camera_azimuth_+5", "camera_azimuth_-5", "camera_azimuth_+10", "camera_azimuth_-10",
              "camera_elevation_+5", "camera_elevation_-5", "light_dome_0.5", "light_dome_1.5", "light_shadow_on"):
        assert n in names, n


@pytest.mark.parametrize("cond", IMAGE_CONDS, ids=lambda c: c.name)
def test_image_perturbation_shape_dtype_determinism_identity(cond):
    x = frames()
    st = rb.episode_state(cond, 4, x.shape[1:], x.device, gen(3))
    y = perturb_images(x, cond, st, gen())
    assert y.shape == x.shape and y.dtype == torch.uint8
    assert not torch.equal(y, x), "the perturbation must change the image"
    # same seed -> same result (episode state and per-step noise)
    st2 = rb.episode_state(cond, 4, x.shape[1:], x.device, gen(3))
    assert torch.equal(perturb_images(x, cond, st2, gen()), y)
    # severity 0 -> exactly the input
    zero = dataclasses.replace(cond, severity=0.0)
    assert torch.equal(perturb_images(x, zero, rb.episode_state(zero, 4, x.shape[1:], x.device, gen(3)), gen()), x)


def test_image_perturbations_do_what_they_say():
    x = frames()
    by = rb.BY_NAME
    y = perturb_images(x, by["view_drop_left"])
    assert (y[:, 0] == 0).all() and torch.equal(y[:, 1], x[:, 1])
    y = perturb_images(x, by["view_drop_right"])
    assert (y[:, 1] == 0).all() and torch.equal(y[:, 0], x[:, 0])
    y = perturb_images(x, by["view_duplicate_left"])
    assert torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, 1], x[:, 0])
    y = perturb_images(x, by["view_swap"])
    assert torch.equal(y[:, 0], x[:, 1]) and torch.equal(y[:, 1], x[:, 0])
    # noise std ~ sigma * 255 where nothing clips
    mid = torch.full((64, 2, 3, 32, 32), 128, dtype=torch.uint8)
    d = perturb_images(mid, by["gaussian_noise_0.05"], gen=gen()).float() - 128
    assert abs(float(d.std()) - 0.05 * 255) < 0.3
    # occlusion: one grey square of the right area, fixed for the episode; one view or both
    for name, frac, views in (("occlusion_one_25", 0.25, 1), ("occlusion_both_40", 0.40, 2)):
        c = by[name]
        st = rb.episode_state(c, 4, x.shape[1:], x.device, gen())
        m = st["occluder"][:, :, 0]  # (N, V, H, W)
        assert ((m.sum(dim=(-2, -1)) > 0).sum(dim=1) == views).all()
        side = round((frac * 32 * 32) ** 0.5)
        assert (m.sum(dim=(-2, -1))[m.sum(dim=(-2, -1)) > 0] == side * side).all()
        y = perturb_images(x, c, st)
        assert (y[:, :, 0][m] == 128).all() and torch.equal(y[:, :, 0][~m], x[:, :, 0][~m])
    # brightness / contrast / hue / blur
    flat = torch.full((1, 2, 3, 8, 8), 100, dtype=torch.uint8)
    assert (perturb_images(flat, by["brightness_1.4"]) == 140).all()
    assert (perturb_images(flat, by["brightness_0.6"]) == 60).all()
    assert torch.equal(perturb_images(flat, by["contrast_0.6"]), flat)  # a flat image has no contrast to lose
    assert float(perturb_images(x, by["contrast_0.6"]).float().std()) < float(x.float().std())
    grey = torch.full((1, 2, 3, 8, 8), 77, dtype=torch.uint8)
    assert torch.equal(perturb_images(grey, by["hue_shift_0.10"]), grey)  # greys stay grey
    red = torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8)
    red[:, :, 0] = 200
    assert perturb_images(red, by["hue_shift_0.10"])[:, :, 1].float().mean() > 20  # red turns towards yellow
    assert torch.equal(perturb_images(flat, by["blur_2"]), flat)  # blur keeps flat regions (reflected borders)
    assert float(perturb_images(x, by["blur_2"]).float().std()) < float(perturb_images(x, by["blur_1"]).float().std())


def test_temporal_perturbations():
    stack = torch.arange(2 * 3).reshape(2, 3, 1, 1, 1, 1).expand(2, 3, 2, 3, 4, 4).to(torch.uint8)
    out = drop_frames(stack, 1)
    assert torch.equal(out[:, 0], stack[:, 0]) and torch.equal(out[:, 1], stack[:, 2]) and torch.equal(out[:, 2], stack[:, 2])
    assert drop_frames(stack, 0) is stack
    # frozen frames: the stack receives a new image only every 2nd step
    seq = [torch.full((2, 2, 3, 4, 4), 10 * (t + 1), dtype=torch.uint8) for t in range(6)]
    no_done = torch.zeros(2, dtype=torch.bool)
    pipe = PerturbedObservations(rb.BY_NAME["frozen_frames_2"], 2, 3, (2, 3, 4, 4), "cpu", gen())
    pipe.reset(seq[0])
    newest = [int(pipe.step(o, no_done)[0, -1, 0, 0, 0, 0]) for o in seq[1:]]
    assert newest == [10, 30, 30, 50, 50]
    # identity at severity 0: plain frame stacking
    for name in ("frozen_frames_2", "frame_drop"):
        c = dataclasses.replace(rb.BY_NAME[name], severity=0.0)
        pipe, ref = PerturbedObservations(c, 2, 3, (2, 3, 4, 4), "cpu", gen()), rb.FrameStacker(2, 3, (2, 3, 4, 4), "cpu")
        assert torch.equal(pipe.reset(seq[0]), ref.reset(seq[0]))
        for o in seq[1:]:
            assert torch.equal(pipe.step(o, no_done), ref.step(o, no_done))
    # frame drop: every stack the policy gets has the newest frame twice
    pipe = PerturbedObservations(rb.BY_NAME["frame_drop"], 2, 3, (2, 3, 4, 4), "cpu", gen())
    pipe.reset(seq[0])
    for o in seq[1:]:
        s = pipe.step(o, no_done)
        assert torch.equal(s[:, -1], o) and torch.equal(s[:, -2], o)


def test_suite_never_touches_training_and_uses_identical_episodes(tmp_path):
    from config import RobustConfig
    from tests.test_mvmae_agent import small_agent
    from tests.test_train_loop import FakeStereoEnv

    env = FakeStereoEnv(num_envs=4, size=32, episode=12)
    agent = small_agent(proprio_dim=3)
    torch.manual_seed(123)
    np.random.seed(5)
    random.seed(6)
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(4, 7))
    counter = env.steps
    env.reset_log.clear()
    states = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    params = [p.detach().clone() for p in agent.actor.parameters()]
    conds = rb.resolve_conditions(["view_drop", "gaussian_noise", "occlusion_one", "temporal", "camera_azimuth",
                                   "camera_elevation", "light_intensity", "shadow_light"])
    res = rb.run_suite(env, agent, 0, RobustConfig(seed=11), conds, ["gaussian_noise_0.10", "frame_drop"], episodes=6,
                       video_dir=tmp_path / "videos", video_scale=1, log=lambda s: None)
    # random streams, env counters and parameters unchanged
    assert torch.equal(torch.get_rng_state(), states[0]) and np.array_equal(np.random.get_state()[1], states[1])
    assert random.getstate() == states[2]
    assert env.steps == counter and env.light == 1.0 and env.cam_shift == 0
    assert all(torch.equal(a, b) for a, b in zip(params, agent.actor.parameters()))
    c = res["conditions"]
    assert res["rounds"] == 2 and res["episodes_per_condition"] == 6
    assert len(c["nominal"]["episodes"]["success"]) == 6
    for name in ("view_drop_left", "gaussian_noise_0.20", "occlusion_one_40", "frame_drop", "light_dome_0.5",
                 "camera_azimuth_+10"):
        assert c[name]["skipped"] is None, name
        for k in rb.METRICS:
            assert 0.0 <= c[name]["success"] <= 1.0 and f"{k}_rel" in c[name]
    assert "fake env cannot tilt" in c["camera_elevation_+5"]["skipped"]
    assert "no visible change" in c["light_shadow_on"]["skipped"]  # applied but invisible -> not reported
    assert c["light_dome_0.5"]["image_change"] > rb.SIM_CHANGE_MIN
    assert res["restore_check_image_change"] < rb.SIM_CHANGE_MIN
    # the same needle placements for every condition: every reset of the suite is round 0's or round 1's placement
    # (the fake env draws them from the global random stream, which the suite seeds per round)
    log = env.reset_log
    assert not torch.equal(log[0], log[1])
    assert all(torch.equal(p, log[0]) or torch.equal(p, log[1]) for p in log)
    ran = sum(e["skipped"] is None for e in c.values())
    assert sum(torch.equal(p, log[0]) for p in log) == ran + 2  # + the invisible shadow light's round 0 + restore check
    # videos of the perturbed input; temporal ones show the whole frame stack
    assert (tmp_path / "videos" / "robust_gaussian_noise_0.10_0.mp4").exists()
    assert (tmp_path / "videos" / "robust_frame_drop_0.mp4").exists()
    # deterministic: the same suite again gives the same numbers
    again = rb.run_suite(env, agent, 0, RobustConfig(seed=11), conds, [], episodes=6, log=lambda s: None)
    assert again["conditions"]["gaussian_noise_0.20"]["episodes"] == c["gaussian_noise_0.20"]["episodes"]
    # outputs
    p = rb.save_json(res, tmp_path / "robust_0.json")
    loaded = json.loads(p.read_text())
    assert loaded["conditions"]["camera_elevation_+5"]["skipped"]
    curves = rb.family_curves(res)
    assert curves["gaussian_noise"][0] == [0.0, 0.05, 0.10, 0.20] and curves["camera_azimuth"][0] == [-10, -5, 0, 5, 10]
    assert "view_drop" not in curves and "camera_elevation" not in curves
    files = rb.save_figures({"MV-MAE": res, "CNN": again}, tmp_path / "fig", "robust")
    assert {f.name for f in files} == {"robust_curves.png", "robust_curves.pdf", "robust_bars.png", "robust_bars.pdf"}
    assert rb.summary_lines(res)[0].startswith("robustness at 0 env steps")
