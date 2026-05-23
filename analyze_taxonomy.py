"""Analyze taxonomy_reports/*.json -- predictive features for steering success.

Given the per-task TaskReport JSONs from `steering_taxonomy.run`, this:

  1. Loads every report into a flat table.
  2. Prints a markdown cross-task summary (geometric + effect columns).
  3. Computes the "predictive rule": does the geometric signature
     (split-half cosine + per-pair variance) classify the effect signature
     (steering moves the metric vs not)?
  4. Reports the rule's accuracy on the 12 tasks and on the held-out
     borderline subset; emits a JSON + CSV with the cross-task table.

Usage:
    python analyze_taxonomy.py                              # reads taxonomy_reports/
    python analyze_taxonomy.py --reports-dir my_reports
    python analyze_taxonomy.py --stab-thresh 0.7 --var-thresh 0.4 \\
        --effect-thresh 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Default thresholds for the predictive rule. These are intentionally
# adjustable: we'll tune on the behavioral + content-specific tasks
# (the "training" set with strong priors) and report rule accuracy on
# the borderline tasks (the "validation" set with weak priors).
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    stab_thresh=0.7,        # split_half_cos_mean: above -> direction is stable
    var_thresh=0.5,         # per_pair_variance: below -> per-pair directions agree
    effect_thresh=0.05,     # |steered - baseline|: above -> measurable effect
    random_thresh=0.05,     # |steered - random_mean|: above -> effect not random
)

BEHAVIORAL = {"refusal", "honesty", "sycophancy", "sentiment", "truthfulness"}
CONTENT_SPECIFIC = {"rag_distractor", "fact_override", "topic_suppression"}
BORDERLINE = {"context_faithfulness", "persona", "politeness", "hallucination_grounding"}


# ---------------------------------------------------------------------------
# Per-task summaries
# ---------------------------------------------------------------------------

def load_reports(reports_dir):
    rows = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def geometric_signature(row, stab_thresh, var_thresh):
    """Classify the direction's geometric structure."""
    cos = row["split_half_cos_mean"]
    var = row["per_pair_variance"]
    if cos >= stab_thresh and var <= var_thresh:
        return "stable"
    if cos < 0.4 or var > 0.7:
        return "unstable"
    return "borderline"


def effect_signature(row, effect_thresh, random_thresh):
    """Classify the empirical effect of the steering intervention."""
    d_base = row["delta_vs_baseline"]
    d_rand = row["delta_vs_random"]
    if abs(d_base) >= effect_thresh and abs(d_rand) >= random_thresh:
        return "steers"
    if abs(d_base) < effect_thresh / 2:
        return "no_effect"
    return "random_explains_it"


def predicted_steers(row, stab_thresh, var_thresh):
    """The rule's prediction: stable geometry -> steering should succeed."""
    return geometric_signature(row, stab_thresh, var_thresh) == "stable"


# ---------------------------------------------------------------------------
# Cross-task table + rule evaluation
# ---------------------------------------------------------------------------

def print_table(rows, args):
    headers = ["task", "kind", "n_pairs", "n_eval", "cos_mean", "cos_std",
               "var", "base", "steered", "rand_mean", "d_vs_base",
               "d_vs_rand", "geom", "effect", "rule_pred"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        geom = geometric_signature(r, args.stab_thresh, args.var_thresh)
        eff = effect_signature(r, args.effect_thresh, args.random_thresh)
        pred = predicted_steers(r, args.stab_thresh, args.var_thresh)
        line = [
            r["task_name"],
            r["task_kind"],
            str(r["n_pairs"]),
            str(r["n_eval"]),
            f'{r["split_half_cos_mean"]:.3f}',
            f'{r["split_half_cos_std"]:.3f}',
            f'{r["per_pair_variance"]:.3f}',
            f'{r["baseline_metric"]:.3f}',
            f'{r["steered_metric"]:.3f}',
            f'{r["random_metric_mean"]:.3f}',
            f'{r["delta_vs_baseline"]:+.3f}',
            f'{r["delta_vs_random"]:+.3f}',
            geom,
            eff,
            "steers" if pred else "no",
        ]
        print("| " + " | ".join(line) + " |")


def rule_accuracy(rows, args, subset=None):
    """% of `subset` rows where the geometric prediction matches the observed effect."""
    if subset is not None:
        rows = [r for r in rows if r["task_name"] in subset]
    if not rows:
        return None
    correct = 0
    for r in rows:
        pred = predicted_steers(r, args.stab_thresh, args.var_thresh)
        eff = effect_signature(r, args.effect_thresh, args.random_thresh)
        observed_steers = (eff == "steers")
        if pred == observed_steers:
            correct += 1
    return correct, len(rows), correct / len(rows)


def write_csv(rows, args, out_path):
    cols = ["task_name", "task_kind", "n_pairs", "n_eval", "target_layer",
            "split_half_cos_mean", "split_half_cos_std", "per_pair_variance",
            "baseline_metric", "steered_metric", "random_metric_mean",
            "random_metric_std", "delta_vs_baseline", "delta_vs_random",
            "seconds", "geometric_signature", "effect_signature",
            "rule_predicts_steers"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            geom = geometric_signature(r, args.stab_thresh, args.var_thresh)
            eff = effect_signature(r, args.effect_thresh, args.random_thresh)
            pred = predicted_steers(r, args.stab_thresh, args.var_thresh)
            w.writerow([r.get(c, "") for c in cols[:15]]
                       + [geom, eff, "yes" if pred else "no"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="taxonomy_reports")
    parser.add_argument("--out-csv", default="taxonomy_summary.csv")
    parser.add_argument("--stab-thresh", type=float, default=DEFAULTS["stab_thresh"])
    parser.add_argument("--var-thresh", type=float, default=DEFAULTS["var_thresh"])
    parser.add_argument("--effect-thresh", type=float, default=DEFAULTS["effect_thresh"])
    parser.add_argument("--random-thresh", type=float, default=DEFAULTS["random_thresh"])
    args = parser.parse_args()

    rows = load_reports(args.reports_dir)
    if not rows:
        raise SystemExit(f"No reports found in {args.reports_dir}/")

    print(f"# Steering taxonomy: {len(rows)} task report(s)\n")
    print(f"Thresholds: stab>={args.stab_thresh}  var<={args.var_thresh}  "
          f"effect>={args.effect_thresh}  random>={args.random_thresh}\n")
    print_table(rows, args)
    print()

    print("## Predictive-rule accuracy\n")
    print("| subset | correct / total | accuracy |")
    print("|---|---|---|")
    for label, subset in [
        ("ALL", None),
        ("behavioral", BEHAVIORAL),
        ("content_specific", CONTENT_SPECIFIC),
        ("borderline (held-out)", BORDERLINE),
    ]:
        result = rule_accuracy(rows, args, subset=subset)
        if result is None:
            print(f"| {label} | 0 / 0 | n/a |")
            continue
        correct, total, acc = result
        print(f"| {label} | {correct} / {total} | {acc:.1%} |")
    print()

    write_csv(rows, args, args.out_csv)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
