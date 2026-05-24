"""Compute correlations between geometric features and steering effect.

For each model, compute:
  Pearson r and Spearman rho between (cos, var) and |delta_vs_random|.

This gives reviewers a single-number summary of how strongly the
geometric features predict empirical effect size, beyond the binary
rule-classification numbers in the main paper.

Usage:
    python correlation_analysis.py                       # all available models
    python correlation_analysis.py --reports taxonomy_reports taxonomy_reports_qwen
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs, ys):
    """Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vs):
        # Average ranks for ties.
        order = sorted(range(n), key=lambda i: vs[i])
        rnk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                rnk[order[k]] = avg
            i = j + 1
        return rnk

    rx = ranks(xs)
    ry = ranks(ys)
    return pearson(rx, ry)


def load_reports(reports_dir):
    rows = []
    for p in sorted(glob.glob(str(Path(reports_dir) / "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def analyze(rows, tag):
    cos = [r["split_half_cos_mean"] for r in rows]
    var = [r["per_pair_variance"] for r in rows]
    d_base = [abs(r["delta_vs_baseline"]) for r in rows]
    d_rand = [abs(r["delta_vs_random"]) for r in rows]
    print(f"\n=== {tag}  (N={len(rows)} tasks) ===")
    print(f"{'feature':>20s}  {'Pearson r':>10s}  {'Spearman rho':>13s}")
    for fname, fvals in [("cos", cos), ("var", var)]:
        for target, tvals in [("|Δ vs baseline|", d_base),
                              ("|Δ vs random|", d_rand)]:
            r = pearson(fvals, tvals)
            rho = spearman(fvals, tvals)
            print(f"{fname:>10s} vs {target:>10s}  {r:>10.3f}  {rho:>13.3f}")
    # Joint signal: cos*var as a single composite predictor
    cv = [c * v for c, v in zip(cos, var)]
    for target, tvals in [("|Δ vs baseline|", d_base),
                          ("|Δ vs random|", d_rand)]:
        r = pearson(cv, tvals)
        rho = spearman(cv, tvals)
        print(f"{'cos*var':>10s} vs {target:>10s}  {r:>10.3f}  {rho:>13.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", nargs="+",
                   default=["taxonomy_reports", "taxonomy_reports_qwen"])
    args = p.parse_args()

    for d in args.reports:
        path = Path(d)
        if not path.exists():
            print(f"\n=== {d} (missing) ===")
            continue
        analyze(load_reports(d), d)


if __name__ == "__main__":
    main()
