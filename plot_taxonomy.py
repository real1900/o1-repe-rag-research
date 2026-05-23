"""Generate the steering-taxonomy paper's main figure(s).

Two-panel figure (taxonomy_figure.pdf / .png):

  A. Geometric characterization scatter
     x = per-pair variance
     y = split-half cosine
     color = hypothesized kind (behavioral, content_specific, borderline)
     marker = observed effect (filled = steers, hollow = no_effect)
     The proposed predictive-rule decision boundary is drawn as a dashed
     box: cos >= stab_thresh AND var <= var_thresh -> "should steer".

  B. Empirical effect bar chart
     One bar per task showing |delta_vs_random|, ordered by hypothesized
     kind. The dashed line is the effect_thresh used to classify "steers"
     vs "no_effect."

Usage:
    python plot_taxonomy.py                        # default reports + thresholds
    python plot_taxonomy.py --reports-dir my_dir --out-prefix paper_fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


KIND_COLORS = {
    "behavioral": "#1b9e77",       # green
    "content_specific": "#d95f02",  # orange
    "borderline": "#7570b3",        # purple
}

DEFAULTS = dict(
    stab_thresh=0.7,
    var_thresh=0.5,
    effect_thresh=0.05,
)


def load_reports(reports_dir):
    rows = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def steers(row, effect_thresh):
    """Empirical: did steering actually move the metric?"""
    return abs(row["delta_vs_random"]) >= effect_thresh and \
        abs(row["delta_vs_baseline"]) >= effect_thresh


def make_figure(rows, args):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: geometric scatter -----------------------------------------
    for row in rows:
        kind = row["task_kind"]
        x = row["per_pair_variance"]
        y = row["split_half_cos_mean"]
        is_steers = steers(row, args.effect_thresh)
        ax_a.scatter(
            x, y,
            s=120,
            color=KIND_COLORS.get(kind, "gray"),
            edgecolors="black",
            linewidths=1.2,
            facecolors=(KIND_COLORS.get(kind, "gray") if is_steers else "white"),
            zorder=3,
        )
        # Annotation -- task name to the right of marker
        ax_a.annotate(
            row["task_name"],
            xy=(x, y), xytext=(6, -2),
            textcoords="offset points",
            fontsize=8, alpha=0.9,
        )

    # Decision boundary box (the "predicted to steer" region)
    rect = mpatches.Rectangle(
        (0, args.stab_thresh),
        args.var_thresh, 1.0 - args.stab_thresh,
        linewidth=1.2, edgecolor="gray", facecolor="lightgray",
        alpha=0.25, linestyle="--", zorder=1,
    )
    ax_a.add_patch(rect)
    ax_a.text(args.var_thresh / 2, args.stab_thresh + (1 - args.stab_thresh) / 2,
              "rule: predicted to steer",
              ha="center", va="center", fontsize=8, color="gray", style="italic")

    ax_a.set_xlim(-0.02, 1.0)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_xlabel("per-pair variance  (0 = aligned; 1 = orthogonal)")
    ax_a.set_ylabel("split-half cosine  (1 = stable direction)")
    ax_a.set_title("A. Geometric characterization")
    ax_a.grid(alpha=0.3)

    # Legend: kinds (colors) + effect (filled / hollow)
    kind_handles = [mpatches.Patch(color=c, label=k) for k, c in KIND_COLORS.items()]
    ax_a.legend(handles=kind_handles, loc="lower left", fontsize=8, framealpha=0.9)

    # --- Panel B: empirical effect bars -------------------------------------
    # Group rows by kind for layout
    by_kind = {"behavioral": [], "content_specific": [], "borderline": []}
    for r in rows:
        by_kind.setdefault(r["task_kind"], []).append(r)

    xs, heights, colors, labels = [], [], [], []
    cursor = 0
    for kind, color in KIND_COLORS.items():
        for r in by_kind.get(kind, []):
            xs.append(cursor)
            heights.append(abs(r["delta_vs_random"]))
            colors.append(color)
            labels.append(r["task_name"])
            cursor += 1
        cursor += 0.5  # gap between groups

    ax_b.bar(xs, heights, color=colors, edgecolor="black", linewidth=1.0)
    ax_b.axhline(args.effect_thresh, color="gray", linestyle="--", linewidth=1.0,
                 label=f"effect threshold ({args.effect_thresh:.2f})")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_b.set_ylabel("|Δ vs random direction|  (effect size)")
    ax_b.set_title("B. Empirical steering effect")
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Steering taxonomy: geometric stability predicts steering success",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="taxonomy_reports")
    parser.add_argument("--out-prefix", default="taxonomy_figure")
    parser.add_argument("--stab-thresh", type=float, default=DEFAULTS["stab_thresh"])
    parser.add_argument("--var-thresh", type=float, default=DEFAULTS["var_thresh"])
    parser.add_argument("--effect-thresh", type=float, default=DEFAULTS["effect_thresh"])
    args = parser.parse_args()

    rows = load_reports(args.reports_dir)
    if not rows:
        raise SystemExit(f"No reports found in {args.reports_dir}/")

    fig = make_figure(rows, args)
    for ext in ("pdf", "png"):
        out = f"{args.out_prefix}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
