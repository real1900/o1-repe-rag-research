"""Leave-one-task-out cross-validation of the predictive rule.

Addresses the reviewer concern: ``the thresholds (tau_stab, tau_var) are
tuned on the same 12 tasks the rule is evaluated on.''

For each held-out task t:
  1. Sweep (tau_stab, tau_var) on the OTHER 11 tasks to find the
     best in-sample accuracy.
  2. Apply those thresholds to task t.
  3. Record whether the prediction was correct.

Repeat for all 12 tasks per model. Reports:
  - Per-fold result table (held-out task, chosen thresholds,
    prediction vs observed).
  - LOO accuracy per model = #correct / 12.
  - Threshold stability (mean +- std of chosen thresholds across folds).
  - Pooled LOO across all 4 models (Llama-1B, 3B, Qwen-3B, 7B):
    held out each (model, task) pair, tune on the other 47.

Usage:
    python loo_cv.py
    python loo_cv.py --pooled
    python loo_cv.py --out-json loo_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Same thresholds as analyze_taxonomy.py
EFFECT_THRESH = 0.05
RANDOM_THRESH = 0.05

# Threshold sweep grid (matches sensitivity_analysis.py)
STAB_GRID = [round(0.40 + 0.05 * i, 2) for i in range(13)]   # 0.40 .. 1.00
VAR_GRID = [round(0.00 + 0.05 * i, 2) for i in range(13)]    # 0.00 .. 0.60

MODELS = {
    "llama-3b":   "taxonomy_reports",
    "qwen-3b":    "taxonomy_reports_qwen",
    "llama-1b":   "taxonomy_reports_llama1b",
    "qwen-7b":    "taxonomy_reports_qwen7b",
}

# Tasks with structural asymmetry in their contrast construction:
# every positive prompt includes a prefix (system instruction, persona
# preamble, context block) that the negative omits entirely. The paper
# identifies these as the third-axis exceptions that geometry alone
# cannot detect.
#
# Verified against steering_taxonomy/tasks/*.py:
#   context_faithfulness: positive = SYSTEM + context + Q; negative = Q
#   persona:              positive = "You are X..." + msg; negative = msg
# Other tasks use paired-response contrasts (same prefix, different
# continuation), which is structurally symmetric.
ASYMMETRIC = {"context_faithfulness", "persona"}


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

def predicted_steers(row, stab, var, use_asymmetry=False):
    """Two-feature rule (default): high cos AND high var.

    Three-feature augmented rule (use_asymmetry=True):
        (high cos AND high var) OR (asymmetric contrast)
    where asymmetry is a per-task structural property of the contrast
    construction, known before any evaluation."""
    base = (row["split_half_cos_mean"] >= stab
            and row["per_pair_variance"] >= var)
    if not use_asymmetry:
        return base
    return base or (row["task_name"] in ASYMMETRIC)


def observed_steers(row):
    return (abs(row["delta_vs_baseline"]) >= EFFECT_THRESH
            and abs(row["delta_vs_random"]) >= RANDOM_THRESH)


def best_thresholds(rows, use_asymmetry=False):
    """Sweep grid, return (tau_stab, tau_var, n_correct, n) that maximizes
    in-sample accuracy. Ties broken by choosing the centroid of the
    plateau (medians of all (stab, var) pairs achieving the max)."""
    best_acc = -1
    plateau = []
    for s in STAB_GRID:
        for v in VAR_GRID:
            c = sum(predicted_steers(r, s, v, use_asymmetry) == observed_steers(r)
                    for r in rows)
            if c > best_acc:
                best_acc = c
                plateau = [(s, v)]
            elif c == best_acc:
                plateau.append((s, v))
    s_med = statistics.median([p[0] for p in plateau])
    v_med = statistics.median([p[1] for p in plateau])
    return s_med, v_med, best_acc, len(rows), len(plateau)


# ---------------------------------------------------------------------------
# LOO
# ---------------------------------------------------------------------------

def load_reports(reports_dir):
    rows = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        with open(p) as f:
            row = json.load(f)
            row["_model"] = reports_dir
            rows.append(row)
    return rows


def loo_cv(rows, label="model", use_asymmetry=False):
    """Leave-one-out CV. Returns dict with fold table + summary."""
    folds = []
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        s, v, train_correct, train_n, plateau_size = best_thresholds(
            train, use_asymmetry)
        pred = predicted_steers(held, s, v, use_asymmetry)
        obs = observed_steers(held)
        folds.append(dict(
            held_out=held["task_name"],
            tau_stab=s,
            tau_var=v,
            train_acc=train_correct / train_n,
            plateau_size=plateau_size,
            cos=round(held["split_half_cos_mean"], 3),
            var=round(held["per_pair_variance"], 3),
            d_rand=round(held["delta_vs_random"], 3),
            predicted_steers=pred,
            observed_steers=obs,
            correct=(pred == obs),
        ))

    n_correct = sum(f["correct"] for f in folds)
    n = len(folds)
    s_vals = [f["tau_stab"] for f in folds]
    v_vals = [f["tau_var"] for f in folds]

    return dict(
        label=label,
        loo_correct=n_correct,
        loo_total=n,
        loo_accuracy=n_correct / n,
        tau_stab_mean=statistics.mean(s_vals),
        tau_stab_std=statistics.stdev(s_vals) if n > 1 else 0.0,
        tau_var_mean=statistics.mean(v_vals),
        tau_var_std=statistics.stdev(v_vals) if n > 1 else 0.0,
        folds=folds,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(result):
    print(f"\n### {result['label']}\n")
    print(f"LOO accuracy: **{result['loo_correct']}/{result['loo_total']} "
          f"= {result['loo_accuracy']:.1%}**")
    print(f"Chosen thresholds across folds: "
          f"tau_stab = {result['tau_stab_mean']:.3f} +- {result['tau_stab_std']:.3f}, "
          f"tau_var = {result['tau_var_mean']:.3f} +- {result['tau_var_std']:.3f}")
    print()
    print("| held-out task | tau_stab | tau_var | train acc | cos | var | "
          "|d_rand| | pred | obs | ok |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in result["folds"]:
        ok = "OK" if f["correct"] else "X"
        print(f"| {f['held_out']} | {f['tau_stab']:.2f} | {f['tau_var']:.2f} "
              f"| {f['train_acc']:.1%} | {f['cos']:.3f} | {f['var']:.3f} "
              f"| {abs(f['d_rand']):.3f} "
              f"| {'steers' if f['predicted_steers'] else 'no'} "
              f"| {'steers' if f['observed_steers'] else 'no'} | {ok} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="loo_cv_results.json")
    ap.add_argument("--pooled", action="store_true",
                    help="Also run pooled LOO across all 4 models (48 folds)")
    ap.add_argument("--asymmetry", action="store_true",
                    help="Use the augmented 3-feature rule "
                         "(cos AND var) OR asymmetric")
    args = ap.parse_args()

    use_asym = args.asymmetry
    all_results = {"rule": "asymmetry-augmented" if use_asym else "two-axis"}
    if use_asym:
        print(f"# Augmented rule: (cos>=tau_stab AND var>=tau_var) "
              f"OR (task in {sorted(ASYMMETRIC)})\n")

    # Per-model LOO
    for model_label, reports_dir in MODELS.items():
        if not Path(reports_dir).exists():
            print(f"# Skipping {model_label}: {reports_dir}/ missing")
            continue
        rows = load_reports(reports_dir)
        if len(rows) != 12:
            print(f"# Warning: {model_label} has {len(rows)} reports, expected 12")
        result = loo_cv(rows, label=f"LOO CV: {model_label} (N={len(rows)})",
                        use_asymmetry=use_asym)
        all_results[model_label] = result
        print_summary(result)

    # Pooled LOO (each (model, task) is one fold; train on the other 47)
    pooled_rows = []
    for model_label, reports_dir in MODELS.items():
        if Path(reports_dir).exists():
            for r in load_reports(reports_dir):
                r["_model_label"] = model_label
                pooled_rows.append(r)
    if pooled_rows:
        pooled = loo_cv(pooled_rows,
                        label=f"LOO CV: pooled across 4 models (N={len(pooled_rows)})",
                        use_asymmetry=use_asym)
        all_results["pooled"] = pooled
        print_summary(pooled)

    # Cross-model generalization: train on 3 models, test on the 4th
    cross_model = {}
    model_names = list(MODELS.keys())
    for held_model in model_names:
        train_rows = []
        for m in model_names:
            if m == held_model:
                continue
            d = MODELS[m]
            if Path(d).exists():
                train_rows.extend(load_reports(d))
        if not Path(MODELS[held_model]).exists():
            continue
        test_rows = load_reports(MODELS[held_model])
        s, v, train_c, train_n, plateau = best_thresholds(train_rows, use_asym)
        test_c = sum(predicted_steers(r, s, v, use_asym) == observed_steers(r)
                     for r in test_rows)
        cross_model[held_model] = dict(
            held_model=held_model,
            train_n=train_n,
            train_acc=train_c / train_n,
            tau_stab=s,
            tau_var=v,
            test_n=len(test_rows),
            test_correct=test_c,
            test_acc=test_c / len(test_rows),
        )
    if cross_model:
        print("\n### Cross-model CV: train on 3 models, test on the 4th\n")
        print("| held-out model | train N | train acc | tau_stab | tau_var "
              "| test N | test correct | test acc |")
        print("|---|---|---|---|---|---|---|---|")
        for m, r in cross_model.items():
            print(f"| {m} | {r['train_n']} | {r['train_acc']:.1%} "
                  f"| {r['tau_stab']:.2f} | {r['tau_var']:.2f} "
                  f"| {r['test_n']} | {r['test_correct']} | {r['test_acc']:.1%} |")
        all_results["cross_model"] = cross_model

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
