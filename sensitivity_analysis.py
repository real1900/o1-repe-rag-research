"""Threshold sensitivity analysis for the steering-taxonomy predictive rule.

The paper's headline rule is `cos >= 0.7 AND var >= 0.3 -> steers`, which
achieves 9/12 (75%) on the Llama-3.2-3B-Instruct corpus. A reviewer will
ask whether that 75% is robust to the choice of thresholds.

This script:
  1. Sweeps tau_stab in [0.5, 0.95] (step 0.05) and tau_var in [0.0, 0.6]
     (step 0.05).
  2. For each (tau_stab, tau_var), evaluates the rule against the observed
     steering outcomes from taxonomy_reports/*.json.
  3. Writes a sensitivity_heatmap.pdf + .png and a sensitivity_table.csv.

The figure makes the robustness claim visually: if there is a sizable
plateau of high accuracy, the chosen thresholds are not overfit to noise.

Usage:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --reports-dir my_reports --effect-thresh 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_reports(reports_dir):
    rows = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def observed_steers(row, effect_thresh):
    return (abs(row["delta_vs_baseline"]) >= effect_thresh
            and abs(row["delta_vs_random"]) >= effect_thresh)


def rule_predicts_steers(row, tau_stab, tau_var):
    return (row["split_half_cos_mean"] >= tau_stab
            and row["per_pair_variance"] >= tau_var)


def accuracy_at(rows, tau_stab, tau_var, effect_thresh):
    correct = sum(
        rule_predicts_steers(r, tau_stab, tau_var) == observed_steers(r, effect_thresh)
        for r in rows
    )
    return correct / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="taxonomy_reports")
    parser.add_argument("--effect-thresh", type=float, default=0.05)
    parser.add_argument("--out-prefix", default="sensitivity")
    args = parser.parse_args()

    rows = load_reports(args.reports_dir)
    if not rows:
        raise SystemExit(f"No reports in {args.reports_dir}/")

    # Threshold grids
    stab_grid = np.round(np.arange(0.50, 0.96, 0.05), 2)
    var_grid = np.round(np.arange(0.00, 0.61, 0.05), 2)
    acc = np.zeros((len(stab_grid), len(var_grid)))
    for i, tau_stab in enumerate(stab_grid):
        for j, tau_var in enumerate(var_grid):
            acc[i, j] = accuracy_at(rows, tau_stab, tau_var, args.effect_thresh)

    # ------- Figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(acc, origin="lower", aspect="auto",
                   extent=(var_grid[0] - 0.025, var_grid[-1] + 0.025,
                           stab_grid[0] - 0.025, stab_grid[-1] + 0.025),
                   cmap="viridis", vmin=0.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("rule accuracy on 12 tasks")
    # Annotate cells with accuracy
    for i, ts in enumerate(stab_grid):
        for j, tv in enumerate(var_grid):
            ax.text(tv, ts, f"{acc[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color=("white" if acc[i, j] < 0.5 else "black"))

    # Mark the paper's reported thresholds
    ax.scatter([0.30], [0.70], color="red", marker="x", s=120,
               linewidth=2.5, label="paper thresholds (0.70, 0.30)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("per-pair-variance threshold $\\tau_{\\text{var}}$ "
                  "(rule: var $\\geq \\tau_{\\text{var}}$)")
    ax.set_ylabel("split-half-cosine threshold $\\tau_{\\text{stab}}$ "
                  "(rule: cos $\\geq \\tau_{\\text{stab}}$)")
    ax.set_title("Rule-accuracy heatmap over the threshold grid")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{args.out_prefix}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")

    # ------- CSV table -----------------------------------------------------
    csv_out = f"{args.out_prefix}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_stab", "tau_var", "accuracy"])
        for i, ts in enumerate(stab_grid):
            for j, tv in enumerate(var_grid):
                w.writerow([float(ts), float(tv), float(acc[i, j])])
    print(f"Wrote {csv_out}")

    # ------- Report best + the paper's threshold -------------------------
    best = np.unravel_index(np.argmax(acc), acc.shape)
    print(f"\nBest accuracy: {acc[best]:.3f} at "
          f"tau_stab={stab_grid[best[0]]}, tau_var={var_grid[best[1]]}")
    paper_i = np.argmin(np.abs(stab_grid - 0.70))
    paper_j = np.argmin(np.abs(var_grid - 0.30))
    print(f"Paper's thresholds (0.70, 0.30): accuracy={acc[paper_i, paper_j]:.3f}")
    # Plateau description: how many grid cells achieve >= 0.70?
    n_good = int(np.sum(acc >= 0.70))
    n_total = acc.size
    print(f"Cells with accuracy >= 0.70: {n_good}/{n_total} ({n_good/n_total:.1%})")
    n_very_good = int(np.sum(acc >= 0.75))
    print(f"Cells with accuracy >= 0.75: {n_very_good}/{n_total} ({n_very_good/n_total:.1%})")


if __name__ == "__main__":
    main()
