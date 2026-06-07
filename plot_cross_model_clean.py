"""Clean cross-model agreement plot for README + paper.

Design: strip plot, one row per task, four colored markers per row
(one per model). Visually: short tight cluster in each row = all
models agree on that task's geometric feature.

Headline message reads instantly without legend reading:
  "every row's markers stack on top of each other"
  ==> the (cos, sigma) signature is a property of the contrast,
      not of the model.

Output:
  docs/img/cross_model.png   -- 170 DPI raster, for the GitHub README
  cross_model_4models.pdf    -- vector, for the paper builds
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    "Llama-3.2-3B":  ("taxonomy_reports",          "#1f77b4", "o"),
    "Llama-3.2-1B":  ("taxonomy_reports_llama1b",  "#56a3ff", "^"),
    "Qwen2.5-3B":    ("taxonomy_reports_qwen",     "#d62728", "o"),
    "Qwen2.5-7B":    ("taxonomy_reports_qwen7b",   "#ff8c8c", "s"),
}


def load_reports(reports_dir):
    out = {}
    for p in sorted(glob.glob(str(Path(reports_dir) / "*.json"))):
        with open(p) as f:
            r = json.load(f)
            out[r["task_name"]] = r
    return out


def make_strip_panel(ax, feat, all_data, task_order, feature_label):
    """Per-task strip plot: each task row has 4 markers (one per model)."""
    for y_idx, task in enumerate(task_order):
        # light horizontal guide line per task
        ax.axhline(y=y_idx, xmin=0, xmax=1, color="lightgray",
                   linewidth=0.5, zorder=0)
        for model_label, (color, marker, value) in all_data[task].items():
            ax.scatter([value], [y_idx],
                       s=110, color=color, marker=marker,
                       edgecolor="black", linewidths=0.9,
                       alpha=0.92, zorder=3)
    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order, fontsize=10)
    ax.set_xlabel(feature_label, fontsize=11.5)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.7, len(task_order) - 0.3)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # so the visually-most-extreme task is on top


def main():
    # Load all models. Fall back gracefully if some directories are absent.
    available = {}
    for label, (d, color, marker) in MODELS.items():
        if Path(d).exists():
            reps = load_reports(d)
            if reps:
                available[label] = (reps, color, marker)

    # Collect data: per task, per model, the cos and var values.
    all_tasks = sorted({t for label, (reps, _, _) in available.items()
                        for t in reps})

    def collect(feat):
        d = {}
        for task in all_tasks:
            d[task] = {}
            for label, (reps, color, marker) in available.items():
                if task in reps:
                    d[task][label] = (color, marker, reps[task][feat])
        return d

    cos_data = collect("split_half_cos_mean")
    var_data = collect("per_pair_variance")

    # Sort tasks once: by mean cos across models, descending
    # (high agreement tasks at top, fact_override at bottom)
    def mean_cos(task):
        vals = [v[2] for v in cos_data[task].values()]
        return -np.mean(vals) if vals else 0.0
    task_order = sorted(all_tasks, key=mean_cos)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13, 5.2), sharey=True,
        gridspec_kw={"wspace": 0.05})
    make_strip_panel(ax_a, "split_half_cos_mean", cos_data, task_order,
                     r"split-half cosine  $\bar{\cos}$")
    make_strip_panel(ax_b, "per_pair_variance", var_data, task_order,
                     r"per-pair variance  $\sigma$")

    # One shared legend at the bottom -- outside the data
    handles = []
    for label, (color, marker) in [(l, (c, m)) for l, (_, c, m) in MODELS.items()]:
        if label in available:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], marker=marker, color="none",
                                  markerfacecolor=color,
                                  markeredgecolor="black",
                                  markersize=10, label=label))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Cross-model agreement: each task's four markers stack on top of "
        "each other\n"
        r"$\Rightarrow$ the $(\bar{\cos}, \sigma)$ signature is a property "
        r"of the contrast, not of the model",
        fontsize=12, y=1.02)

    Path("docs/img").mkdir(parents=True, exist_ok=True)
    for out in ("docs/img/cross_model.png",
                "cross_model_4models.pdf"):
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
