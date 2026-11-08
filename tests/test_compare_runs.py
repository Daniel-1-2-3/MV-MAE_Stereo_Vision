"""tools/compare_runs.py on synthetic wandb history rows (no network)."""

import json

from tools import compare_runs as cr


def rows(scale: float):
    out = []
    for i, step in enumerate((50_000, 100_000, 150_000, 200_000)):
        out.append({"_step": step, "eval/success": min(1.0, scale * i / 3), "eval/success_any": min(1.0, scale * i / 2),
                    "eval/episode_return": 10.0 * i, "train/ignored": 1.0})
        if step % 100_000 == 0:
            r = {"_step": step, "robust/nominal/success": scale * i / 3, "robust/mean_success": 0.1}
            for name in ("gaussian_noise_0.05", "gaussian_noise_0.20", "view_drop_left", "camera_azimuth_+5"):
                r[f"robust/{name}/success"] = scale * i / 6
                r[f"robust/{name}/success_rel"] = 0.5
            out.append(r)  # logged separately at the same step: merged by step
    return out


def test_compare_runs_figures_from_history_rows(tmp_path):
    runs = {}
    for label, scale in (("MV-MAE", 1.0), ("CNN", 0.6)):
        evals, robust = cr.parse_rows(rows(scale))
        assert [e["step"] for e in evals] == [50_000, 100_000, 150_000, 200_000]
        assert sorted(robust) == [100_000, 200_000]
        c = robust[200_000]["conditions"]
        assert c["gaussian_noise_0.20"]["family"] == "gaussian_noise" and c["gaussian_noise_0.20"]["severity"] == 0.20
        assert "success_rel" not in c["view_drop_left"]
        step = cr.best_robust_step(evals, robust)
        assert step == 200_000
        runs[label] = {"evals": evals, "robust": robust[step]}
    assert cr.steps_to_threshold(runs["MV-MAE"]["evals"], 0.5) == 150_000
    assert cr.steps_to_threshold(runs["CNN"]["evals"], 0.9) is None
    summary = cr.make_figures(runs, tmp_path, 0.5)
    for stem in ("success_vs_steps", "steps_to_threshold", "robustness_curves", "robustness_bars"):
        assert (tmp_path / f"{stem}.png").exists() and (tmp_path / f"{stem}.pdf").exists(), stem
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["runs"]["MV-MAE"]["steps_to_threshold"] == 150_000 and saved["runs"]["CNN"]["steps_to_threshold"] == 200_000
    assert summary["runs"]["CNN"]["robust_step"] == 200_000
    # a run that never reaches the threshold is drawn as "not reached"
    high = cr.make_figures(runs, tmp_path / "high", 0.9)
    assert high["runs"]["MV-MAE"]["steps_to_threshold"] == 200_000 and high["runs"]["CNN"]["steps_to_threshold"] is None
    assert (tmp_path / "high" / "steps_to_threshold.png").exists()
