"""Layer-sweep heatmap: visualize cos and var across (task, layer).

Two heatmaps side-by-side:
  Left:  split-half cosine, rows = tasks, columns = layers (0..n_layers-1)
  Right: per-pair variance, same shape

Tasks are ordered by hypothesized kind for visual grouping. The
data-derived rule shades color: cos in [0,1] uses one colormap;
var in [0,1] uses another. The rule region (high cos + high var)
visually pops as the upper-right of each panel for cos and var
respectively.

Usage:
    python plot_layer_sweep.py
    python plot_layer_sweep.py --input layer_sweep_llama3b.json \\
       --out-prefix layer_sweep_heatmap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Task ordering by hypothesized kind for visual grouping.
TASK_ORDER = [
    # Behavioral
    "refusal", "honesty", "sycophancy", "sentiment", "truthfulness",
    # Content-specific
    "rag_distractor", "fact_override", "topic_suppression",
    # Borderline
    "context_faithfulness", "persona", "politeness",
    "hallucination_grounding",
]

DISPLAY_NAMES = {
    "hallucination_grounding": "halluc.\\_grounding",
    "context_faithfulness": "context\\_faithfulness",
    "rag_distractor": "rag\\_distractor",
    "fact_override": "fact\\_override",
    "topic_suppression": "topic\\_suppression",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="layer_sweep_llama3b.json")
    p.add_argument("--out-prefix", default="layer_sweep_heatmap")
    args = p.parse_args()

    data = json.load(open(args.input))
    n_layers = data["n_layers"]
    layers = sorted(int(l) for l in data["layers"])
    tasks = [t for t in TASK_ORDER if t in data["tasks"]]

    cos_mat = np.zeros((len(tasks), len(layers)))
    var_mat = np.zeros((len(tasks), len(layers)))
    for i, t in enumerate(tasks):
        for j, L in enumerate(layers):
            entry = data["tasks"][t][str(L)]
            cos_mat[i, j] = entry["cos"]
            var_mat[i, j] = entry["var"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title in [(ax_a, cos_mat, "split-half cosine"),
                           (ax_b, var_mat, "per-pair variance")]:
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0.0, vmax=1.0,
                       extent=(layers[0] - 0.5, layers[-1] + 0.5,
                               -0.5, len(tasks) - 0.5))
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([t.replace("_", "\\_") for t in tasks],
                           fontsize=8)
        ax.set_xticks([0, 7, 14, 21, layers[-1]])
        ax.set_xlabel(f"layer (of {n_layers})")
        ax.set_title(title)
        # Mark the layer used in the main paper (layer 7 for Llama-3B)
        ax.axvline(7, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
        # Horizontal separators between hypothesized kinds
        for sep_y in [4.5, 7.5]:
            ax.axhline(sep_y, color="white", linewidth=1.0, alpha=0.6)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"Layer sweep on {data['model_id']}: per-task per-layer "
                 "(cos, var). Red line: main-paper layer choice (=7).",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    for ext, dpi in [("pdf", 200), ("png", 110)]:
        out = f"{args.out_prefix}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
