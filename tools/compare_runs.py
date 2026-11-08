"""Paper figures comparing two training runs (e.g. MV-MAE vs the CNN baseline) from their wandb histories.

    python tools/compare_runs.py --runs <entity>/mvmae-drqv2-surgery/<run id> <entity>/mvmae-drqv2-surgery/<run id> \
        --labels MV-MAE CNN
    # robustness from offline robustness_eval.py results (more episodes) instead of the training-time suite:
    python tools/compare_runs.py --runs <a> <b> --labels MV-MAE CNN \
        --robust_json outputs/robustness/a_ckpt_best/robust.json outputs/robustness/b_ckpt_best/robust.json

Writes PNG + PDF figures to --out (default outputs/compare/):
  success_vs_steps     nominal evaluation success (solid) and success at any point (dashed) vs env steps
  steps_to_threshold   env steps of the first evaluation with success >= --threshold (default 0.5)
  robustness_curves    success vs severity, one panel per perturbation family, at each run's best checkpoint
                       (nominal at severity 0)
  robustness_bars      grouped bars for every condition, run A vs run B, at the best checkpoint
  summary.json         the numbers behind the figures

"Best checkpoint" = the step with a robustness evaluation whose nominal evaluation scored best by (success,
success_any, return), the same order as ckpt_best.pt. Needs `wandb login` (read access to both runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robustness  # noqa: E402

EVAL_KEYS = ("eval/success", "eval/success_any", "eval/episode_return")


def fetch_rows(path: str) -> tuple[str, list[dict]]:
    """(run name, history rows with the eval/ and robust/ scalars, keyed by env step `_step`)."""
    import wandb

    run = wandb.Api(timeout=120).run(path)
    rows = []
    for row in run.scan_history(page_size=1000):
        keep = {k: v for k, v in row.items()
                if k.startswith(("eval/", "robust/")) and isinstance(v, (int, float)) and not isinstance(v, bool)}
        if keep:
            keep["_step"] = int(row["_step"])
            rows.append(keep)
    return run.name, rows


def parse_rows(rows: list[dict]) -> tuple[list[dict], dict[int, dict]]:
    """History rows -> (evaluations sorted by step, {step: robustness results in robustness.run_suite's format})."""
    by_step: dict[int, dict] = {}
    for row in rows:
        by_step.setdefault(int(row["_step"]), {}).update(row)
    evals, robust = [], {}
    for step in sorted(by_step):
        row = by_step[step]
        if "eval/success" in row:
            evals.append({"step": step, **{k.split("/", 1)[1]: row[k] for k in EVAL_KEYS if k in row}})
        conds = {}
        for key, value in row.items():
            parts = key.split("/")
            if len(parts) != 3 or parts[0] != "robust" or parts[2] not in robustness.METRICS:
                continue
            name = parts[1]
            c = robustness.NOMINAL if name == "nominal" else robustness.BY_NAME.get(name)
            if c is None:
                continue
            entry = conds.setdefault(name, {"family": c.family, "severity": c.severity, "kind": c.kind,
                                            "variant": c.variant, "skipped": None})
            entry[parts[2]] = float(value)
        if "nominal" in conds and all("success" in e for e in conds.values()):
            robust[step] = {"env_step": step, "conditions": {"nominal": conds.pop("nominal"), **conds}}
    return evals, robust


def best_robust_step(evals: list[dict], robust: dict[int, dict]) -> int | None:
    if not robust:
        return None
    score = {e["step"]: (e.get("success", 0.0), e.get("success_any", 0.0), e.get("episode_return", 0.0)) for e in evals}
    return max(robust, key=lambda s: score.get(s, (robust[s]["conditions"]["nominal"]["success"], 0.0, 0.0)))


def steps_to_threshold(evals: list[dict], threshold: float) -> int | None:
    return next((e["step"] for e in evals if e.get("success", 0.0) >= threshold), None)


def make_figures(runs: dict[str, dict], out: Path, threshold: float, formats=("png", "pdf")) -> dict:
    """runs: {label: {"evals": [...], "robust": results or None}}. Returns the summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labels = list(runs)
    colors = {lab: robustness.SERIES[i % 2] for i, lab in enumerate(labels)}
    summary = {"threshold": threshold, "runs": {}}

    def save(fig, stem):
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out / f"{stem}.{fmt}", dpi=200)
        plt.close(fig)

    # ---- success vs env steps (nominal evaluation)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for lab in labels:
        ev = runs[lab]["evals"]
        xs = [e["step"] / 1e6 for e in ev]
        ax.plot(xs, [e.get("success", float("nan")) for e in ev], color=colors[lab], linewidth=1.5, label=lab)
        ax.plot(xs, [e.get("success_any", float("nan")) for e in ev], color=colors[lab], linewidth=1.0,
                linestyle="--", alpha=0.8, label=f"{lab} (success at any point)")
    ax.axhline(threshold, color=robustness.INK_2, linewidth=0.6, linestyle=":")
    ax.set_xlabel("environment steps (millions)", fontsize=9)
    ax.set_ylabel("evaluation success", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    robustness.style_axes(ax)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "success_vs_steps")

    # ---- steps to threshold
    fig, ax = plt.subplots(figsize=(4.0, 0.6 + 0.5 * len(labels)))
    max_step = max((e["step"] for r in runs.values() for e in r["evals"]), default=1)
    for i, lab in enumerate(labels):
        s = steps_to_threshold(runs[lab]["evals"], threshold)
        summary["runs"].setdefault(lab, {})["steps_to_threshold"] = s
        if s is None:
            ax.text(0.01 * max_step / 1e6, i, f"not reached (> {max_step / 1e6:.2f} M)", va="center", fontsize=8,
                    color=robustness.INK_2)
        else:
            ax.barh(i, s / 1e6, color=colors[lab], height=0.55)
            ax.text(s / 1e6, i, f"  {s / 1e6:.2f} M", va="center", fontsize=8, color=robustness.INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max_step / 1e6 * 1.25)
    ax.invert_yaxis()
    ax.set_xlabel(f"environment steps (millions) to the first evaluation with success >= {threshold:g}", fontsize=8)
    robustness.style_axes(ax)
    save(fig, "steps_to_threshold")

    # ---- robustness at the best checkpoint
    robust = {lab: r["robust"] for lab, r in runs.items() if r.get("robust")}
    if robust:
        robustness.save_figures(robust, out, "robustness", formats=formats)
    for lab, r in runs.items():
        ev = r["evals"]
        summary["runs"].setdefault(lab, {}).update({
            "final_success": ev[-1].get("success") if ev else None,
            "best_success": max((e.get("success", 0.0) for e in ev), default=None),
            "robust_step": r["robust"]["env_step"] if r.get("robust") else None,
            "robust_success": ({n: e.get("success") for n, e in r["robust"]["conditions"].items()
                                if e.get("skipped") is None} if r.get("robust") else None),
        })
    (out / "summary.json").write_text(json.dumps(robustness._clean(summary), indent=1))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs=2, required=True, help="two wandb run paths <entity>/<project>/<run id>")
    p.add_argument("--labels", nargs=2, default=None, help="default: the run names")
    p.add_argument("--robust_json", nargs=2, default=None, help="robustness_eval.py robust.json per run (optional)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="outputs/compare")
    args = p.parse_args()
    runs = {}
    for i, path in enumerate(args.runs):
        name, rows = fetch_rows(path)
        label = args.labels[i] if args.labels else name
        evals, robust = parse_rows(rows)
        step = best_robust_step(evals, robust)
        res = robust.get(step) if step is not None else None
        if args.robust_json:
            res = json.loads(Path(args.robust_json[i]).read_text())
            for e in res["conditions"].values():  # JSON null -> NaN for the plots
                for k in robustness.METRICS:
                    if k in e and e[k] is None:
                        e[k] = float("nan")
        print(f"{label}: {len(evals)} evaluations, {len(robust)} robustness evaluations"
              + (f", robustness from step {res['env_step']}" if res else ", no robustness data"))
        runs[label] = {"evals": evals, "robust": res}
    summary = make_figures(runs, Path(args.out), args.threshold)
    print(json.dumps(summary["runs"], indent=1, default=str)[:2000])
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
