"""Cross-model agreement plot: shows (cos, var) is a property of the contrast.

Two-panel figure:
  A. Llama-3B (cos) vs Qwen-3B (cos), one point per task, x=y diagonal.
  B. Llama-3B (var) vs Qwen-3B (var), same idea.

Points clustered near the diagonal => the geometric feature is invariant
to model choice. Outliers identify which tasks (if any) have
model-dependent geometry.

Optionally adds Llama-1B and Qwen-7B as additional points per task once
those reports are available.

Usage:
    python plot_cross_model.py
    python plot_cross_model.py --out-prefix cross_model_4_models
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPORT_DIRS = {
    "Llama-3B (layer 7)":  ("taxonomy_reports",          "#1f77b4", "o"),
    "Qwen-3B (layer 9)":   ("taxonomy_reports_qwen",     "#d62728", "o"),
    "Llama-1B (layer 4)":  ("taxonomy_reports_llama1b",  "#1f77b4", "^"),
    "Qwen-7B (layer 7)":   ("taxonomy_reports_qwen7b",   "#d62728", "s"),
}


def load_reports(reports_dir):
    out = {}
    for p in sorted(glob.glob(str(Path(reports_dir) / "*.json"))):
        with open(p) as f:
            r = json.load(f)
            out[r["task_name"]] = r
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="Llama-3B (layer 7)")
    parser.add_argument("--out-prefix", default="cross_model_agreement")
    args = parser.parse_args()

    available = {}
    for label, (d, _, _) in REPORT_DIRS.items():
        if Path(d).exists():
            reps = load_reports(d)
            if reps:
                available[label] = reps
    if args.anchor not in available:
        raise SystemExit(f"anchor {args.anchor!r} not available; "
                         f"have {list(available)}")
    anchor_reps = available[args.anchor]
    others = [k for k in available if k != args.anchor]
    if not others:
        raise SystemExit(f"need at least one non-anchor model; have only "
                         f"{args.anchor}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))
    for ax, feat, name in [(ax_a, "split_half_cos_mean",
                            "split-half cosine"),
                           (ax_b, "per_pair_variance",
                            "per-pair variance")]:
        # Diagonal reference line
        ax.plot([0, 1.05], [0, 1.05], color="gray", linestyle=":",
                linewidth=0.8, zorder=1)
        # Scatter each non-anchor model against the anchor
        for label in others:
            color = REPORT_DIRS[label][1]
            marker = REPORT_DIRS[label][2]
            xs, ys, names = [], [], []
            for task, anchor_row in anchor_reps.items():
                if task not in available[label]:
                    continue
                xs.append(anchor_row[feat])
                ys.append(available[label][task][feat])
                names.append(task)
            ax.scatter(xs, ys, s=80, color=color, marker=marker,
                       edgecolor="black", linewidths=1.0,
                       label=label, alpha=0.85, zorder=3)
            # Annotate each point with task name (small font)
            for xi, yi, ni in zip(xs, ys, names):
                ax.annotate(ni, xy=(xi, yi), xytext=(4, 2),
                            textcoords="offset points",
                            fontsize=7, alpha=0.8)
        # Shade the +-0.05 agreement band around the diagonal
        ax.fill_between([0, 1.05], [-0.05, 1.00], [0.05, 1.10],
                        color="lightgray", alpha=0.35, zorder=0,
                        label="±0.05 band")
        ax.set_xlabel(f"{name} ({args.anchor})")
        ax.set_ylabel(f"{name} (other model)")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        if ax is ax_a:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    fig.suptitle("Cross-model agreement: geometric features depend on "
                 "the contrast, not the model", fontsize=12, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{args.out_prefix}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")

    # Numeric summary
    print(f"\n=== Cross-model agreement summary (anchor: {args.anchor}) ===")
    for label in others:
        diffs_cos, diffs_var = [], []
        for task, anchor_row in anchor_reps.items():
            if task not in available[label]:
                continue
            diffs_cos.append(
                anchor_row["split_half_cos_mean"]
                - available[label][task]["split_half_cos_mean"])
            diffs_var.append(
                anchor_row["per_pair_variance"]
                - available[label][task]["per_pair_variance"])
        n = len(diffs_cos)
        ac = max(abs(x) for x in diffs_cos)
        av = max(abs(x) for x in diffs_var)
        within_05_cos = sum(1 for x in diffs_cos if abs(x) <= 0.05)
        within_05_var = sum(1 for x in diffs_var if abs(x) <= 0.05)
        print(f"  {label:25s}  N={n}  max|Δcos|={ac:.3f}  "
              f"max|Δvar|={av:.3f}  within±0.05: cos={within_05_cos}/{n}, "
              f"var={within_05_var}/{n}")


if __name__ == "__main__":
    main()
