"""Supervised-probe baseline: an alternative predictor of steering success.

Reviewer pushback: ``why use cos + var when a supervised probe on the
contrast pairs would tell you the same thing?'' This script answers
that directly.

For each task:
  1. Build N contrastive pairs (using the task's existing build_pairs).
  2. Extract mean-pooled hidden states at the target layer for the
     positive and negative prompts.
  3. Train a logistic regression probe to discriminate positive from
     negative, using 5-fold cross-validation. Report mean CV accuracy.
  4. Compare to the observed |Delta_rand| from taxonomy_reports/.

The hypothesis under test: high probe accuracy = high cos + high var,
i.e., the probe is redundant with our geometric features. If true,
the geometric features lose to a simpler baseline. If false (probe
saturates near 1.0 on all 12 tasks regardless of whether they steer),
the geometric features capture something a probe cannot.

Usage on M4 (after taxonomy_reports/ exists):

    cd ~/selfrag_run
    source ~/selfrag_env/bin/activate
    python probe_baseline.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --layer 7 --n-pairs 200

Output: probe_baseline.json with per-task probe accuracy + correlation
with the geometric and empirical features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.tasks import ALL_TASKS


def kfold_logreg_accuracy(X, y, k=5, seed=2026):
    """Plain k-fold CV accuracy for a logistic-regression probe.

    Uses scipy/sklearn if available; falls back to a tiny pure-numpy
    closed-form ridge classifier so the script runs even on bare
    environments.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            # Standardize features per fold (fit scaler on train, apply to test).
            # Hidden states from fp16 LLMs have wide-ranging magnitudes that
            # trigger overflow in sklearn's matmul without scaling.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx].astype(np.float32))
            X_test = scaler.transform(X[test_idx].astype(np.float32))
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X_train, y[train_idx])
            accs.append(clf.score(X_test, y[test_idx]))
        return float(np.mean(accs)), float(np.std(accs))
    except ImportError:
        # closed-form ridge classifier (fallback)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, k)
        accs = []
        for i in range(k):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(k) if j != i])
            Xt, yt = X[train], y[train].astype(float) * 2 - 1
            Xs = X[test]
            ys = y[test].astype(float) * 2 - 1
            # ridge: w = (X^T X + lambda I)^-1 X^T y
            lam = 1.0
            A = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
            w = np.linalg.solve(A, Xt.T @ yt)
            preds = np.sign(Xs @ w)
            accs.append(float(np.mean(preds == ys)))
        return float(np.mean(accs)), float(np.std(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--reports-dir", default="taxonomy_reports",
                   help="Used to fetch the observed |delta_rand| for correlation.")
    p.add_argument("--out-json", default="probe_baseline.json")
    args = p.parse_args()

    runner = LlamaModelRunner(model_id=args.model)
    print(f"Probing layer {args.layer} of {args.model}, N={args.n_pairs} pairs/task")

    # Load observed deltas for the correlation analysis
    reports_dir = Path(args.reports_dir)
    observed = {}
    if reports_dir.exists():
        for rj in reports_dir.glob("*.json"):
            with open(rj) as f:
                d = json.load(f)
                observed[d["task_name"]] = d

    results = {}
    for tc in ALL_TASKS:
        task = tc()
        print(f"\n----- {task.name} -----")
        pairs = task.build_pairs(n=args.n_pairs)
        if len(pairs) < 20:
            print(f"  too few pairs ({len(pairs)}), skipping")
            continue

        pos = [p.positive for p in pairs]
        neg = [p.negative for p in pairs]
        # mean_act handles batching internally; do positive and negative
        h_pos = runner.mean_act(pos, args.layer).cpu().numpy()
        h_neg = runner.mean_act(neg, args.layer).cpu().numpy()

        X = np.concatenate([h_pos, h_neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(h_pos), dtype=int),
                            np.zeros(len(h_neg), dtype=int)])

        acc_mean, acc_std = kfold_logreg_accuracy(X, y, k=5, seed=2026)
        obs = observed.get(task.name, {})
        d_rand = obs.get("delta_vs_random", None)
        cos = obs.get("split_half_cos_mean", None)
        var = obs.get("per_pair_variance", None)

        results[task.name] = dict(
            task_name=task.name,
            probe_accuracy=acc_mean,
            probe_accuracy_std=acc_std,
            n_pairs=len(pairs),
            cos=cos,
            var=var,
            delta_vs_random=d_rand,
        )
        print(f"  probe acc: {acc_mean:.3f} +- {acc_std:.3f}  "
              f"|d_rand|: "
              f"{('%.3f' % abs(d_rand)) if d_rand is not None else 'n/a'}")

    # Correlation analysis: does probe accuracy predict |delta_rand|?
    matched = [(r["probe_accuracy"], abs(r["delta_vs_random"]))
               for r in results.values()
               if r["delta_vs_random"] is not None]
    if len(matched) >= 3:
        ps = np.array([m[0] for m in matched])
        ds = np.array([m[1] for m in matched])
        r_pearson = float(np.corrcoef(ps, ds)[0, 1])
    else:
        r_pearson = None

    matched_geo = [(r["cos"] * r["var"], abs(r["delta_vs_random"]))
                   for r in results.values()
                   if all(r[k] is not None for k in
                          ["cos", "var", "delta_vs_random"])]
    if len(matched_geo) >= 3:
        gs = np.array([m[0] for m in matched_geo])
        ds = np.array([m[1] for m in matched_geo])
        r_geo = float(np.corrcoef(gs, ds)[0, 1])
    else:
        r_geo = None

    summary = dict(
        model=args.model,
        layer=args.layer,
        n_pairs_per_task=args.n_pairs,
        per_task=results,
        correlation_probe_vs_delta_rand=r_pearson,
        correlation_cos_x_var_vs_delta_rand=r_geo,
    )

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_json}")
    print(f"Correlation r(probe_acc, |d_rand|):     "
          f"{r_pearson:.3f}" if r_pearson is not None else " n/a")
    print(f"Correlation r(cos*var,   |d_rand|):     "
          f"{r_geo:.3f}" if r_geo is not None else " n/a")
    print("\nInterpretation:")
    print("  If r(probe) > r(cos*var):  probe is a better predictor")
    print("     -- our geometric story loses to a simple baseline")
    print("  If r(probe) ~= r(cos*var): comparable, choose by interp")
    print("  If r(probe) < r(cos*var):  geometric features capture more")
    print("     -- probe accuracy saturates regardless of steering")


if __name__ == "__main__":
    main()
