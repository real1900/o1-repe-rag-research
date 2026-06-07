"""Clean README-style cross-model agreement plot.

The labeled version (`plot_cross_model.py` -> `cross_model_4models.pdf`)
is dense by design so a reader can identify every task in the paper.
That gets unreadable at README width where many points pile up at
cos~=1.0 and var~=0.

This version drops the per-point labels, uses bigger markers, and
labels only `fact_override` (the conspicuous outlier at low cos).
The visual message remains: every point sits on the y=x diagonal.

Output: docs/img/cross_model.png
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPORT_DIRS = {
    "Llama-3B (layer 7)":  ("taxonomy_reports",          "#1f77b4", "o", 110),
    "Qwen-3B (layer 9)":   ("taxonomy_reports_qwen",     "#d62728", "o", 110),
    "Llama-1B (layer 4)":  ("taxonomy_reports_llama1b",  "#1f77b4", "^", 120),
    "Qwen-7B (layer 7)":   ("taxonomy_reports_qwen7b",   "#d62728", "s", 110),
}
ANCHOR = "Llama-3B (layer 7)"
OUTLIER = "fact_override"  # only point worth annotating; lives at low cos


def load_reports(reports_dir):
    out = {}
    for p in sorted(glob.glob(str(Path(reports_dir) / "*.json"))):
        with open(p) as f:
            r = json.load(f)
            out[r["task_name"]] = r
    return out


def main():
    available = {}
    for label, (d, _, _, _) in REPORT_DIRS.items():
        if Path(d).exists():
            reps = load_reports(d)
            if reps:
                available[label] = reps
    anchor_reps = available[ANCHOR]
    others = [k for k in available if k != ANCHOR]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.8))
    outlier_annotated = False  # only annotate fact_override once total
    for ax, feat, name in [(ax_a, "split_half_cos_mean",
                            r"split-half cosine $\bar{\cos}$"),
                           (ax_b, "per_pair_variance",
                            r"per-pair variance $\sigma$")]:
        # Shaded +/- 0.05 band first so it's behind everything
        ax.fill_between([0, 1.05], [-0.05, 1.00], [0.05, 1.10],
                        color="lightgray", alpha=0.4, zorder=0,
                        label=r"$\pm$0.05 agreement band")
        # Diagonal
        ax.plot([0, 1.05], [0, 1.05], color="dimgray",
                linestyle="--", linewidth=1.0, zorder=1, label="y = x")
        # Per-model scatter
        for label in others:
            _, color, marker, size = REPORT_DIRS[label]
            xs, ys, names = [], [], []
            for task, anchor_row in anchor_reps.items():
                if task not in available[label]:
                    continue
                xs.append(anchor_row[feat])
                ys.append(available[label][task][feat])
                names.append(task)
            ax.scatter(xs, ys, s=size, color=color, marker=marker,
                       edgecolor="black", linewidths=0.9,
                       label=label, alpha=0.78, zorder=3)
            # Only label fact_override on the cos panel, only once total
            # across all (model, panel) iterations. It's the visual outlier;
            # everyone else sits on the diagonal.
            if feat == "split_half_cos_mean" and not outlier_annotated:
                for xi, yi, ni in zip(xs, ys, names):
                    if ni == OUTLIER:
                        ax.annotate(ni, xy=(xi, yi),
                                    xytext=(15, 0),
                                    textcoords="offset points",
                                    fontsize=9.5, alpha=0.95,
                                    fontstyle="italic", fontweight="bold")
                        outlier_annotated = True
                        break
        ax.set_xlabel(f"{name} on {ANCHOR}", fontsize=10.5)
        ax.set_ylabel(f"{name} on other model", fontsize=10.5)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal")
        if ax is ax_a:
            ax.legend(loc="lower right", fontsize=8.5, framealpha=0.94)

    fig.suptitle("Cross-model agreement: the geometric features are a "
                 "property of the contrast, not the model",
                 fontsize=12.5, y=1.00, fontweight="bold")
    fig.tight_layout()
    Path("docs/img").mkdir(parents=True, exist_ok=True)
    for out in ("docs/img/cross_model.png",
                "cross_model_4models.pdf"):
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
