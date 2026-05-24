"""Additive-CAA sweep: does (cos, var) predict additive-CAA effects too?

The main paper's predictive rule was trained on ablation effects
(scale-free, projects the direction out). A natural concern: does the
same rule predict effects under additive CAA (Rimsky et al., 2023),
which ADDS alpha*direction to the residual stream?

This script:
  1. Picks a representative subset of tasks covering each regime of the
     predictive rule.
  2. For each task, extracts the (CAA) direction once.
  3. Runs the eval set under baseline, ablation, and additive CAA with
     alpha in {0.5, 1.0, 2.0, 4.0}.
  4. Reports per-task per-alpha metrics.

Output: additive_caa_<model_tag>.json

Usage:
    python additive_caa_sweep.py
    python additive_caa_sweep.py --model Qwen/Qwen2.5-3B-Instruct \\
      --layer 9 --tag qwen3b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.protocol import (
    split_half_cosine,
    per_pair_variance,
)
from steering_taxonomy.tasks import ALL_TASKS


# Representative tasks chosen to span the predictive-rule regimes:
#   refusal              cleanly steers under ablation (hi cos, hi var)
#   rag_distractor       cleanly steers under ablation (hi cos, hi var)
#   honesty              null under ablation (hi cos, ~0 var, symmetric)
#   context_faithfulness exception (hi cos, ~0 var, asymmetric prefix)
DEFAULT_TASKS = ["refusal", "rag_distractor", "honesty", "context_faithfulness"]
DEFAULT_ALPHAS = [0.5, 1.0, 2.0, 4.0]


def _score_set(model_runner, task, examples, hook, layer):
    scores = []
    for ex in examples:
        completion = model_runner.generate(ex.prompt, hook=hook,
                                           layer=layer, max_new_tokens=64)
        scores.append(task.score_completion(completion, ex))
    return float(np.mean(scores))


def evaluate_additive(task, runner, layer, n_pairs, n_eval, alphas, seed=2026):
    pairs = task.build_pairs(n=n_pairs)
    eval_examples = task.build_eval(n=n_eval)
    pos = runner.mean_act([p.positive for p in pairs], layer=layer)
    neg = runner.mean_act([p.negative for p in pairs], layer=layer)
    per_pair = pos - neg
    cos_mean, cos_std = split_half_cosine(per_pair)
    var = per_pair_variance(per_pair)
    direction = per_pair.mean(0)
    direction = direction / (direction.norm() + 1e-9)
    print(f"  [direction] cos={cos_mean:.3f}+-{cos_std:.3f}  var={var:.3f}",
          flush=True)

    t = time.time()
    print("  [eval] baseline...", flush=True)
    baseline = _score_set(runner, task, eval_examples, hook=None, layer=layer)
    print(f"    baseline={baseline:.3f} ({time.time()-t:.0f}s)", flush=True)

    t = time.time()
    print("  [eval] ablation (scale-free)...", flush=True)
    ablate = _score_set(runner, task, eval_examples,
                        hook=("ablate", direction), layer=layer)
    print(f"    ablate={ablate:.3f} ({time.time()-t:.0f}s)", flush=True)

    add_results = {}
    for alpha in alphas:
        t = time.time()
        print(f"  [eval] additive (alpha={alpha})...", flush=True)
        add_score = _score_set(runner, task, eval_examples,
                               hook=("add", direction, alpha), layer=layer)
        add_results[str(alpha)] = add_score
        print(f"    add(alpha={alpha})={add_score:.3f}  "
              f"Δbase={add_score-baseline:+.3f}  "
              f"({time.time()-t:.0f}s)", flush=True)

    return dict(
        task=task.name,
        hypothesized_kind=task.hypothesized_kind,
        layer=layer,
        n_pairs=len(pairs),
        n_eval=len(eval_examples),
        cos_mean=cos_mean, cos_std=cos_std, var=var,
        baseline=baseline,
        ablate=ablate,
        ablate_delta_base=ablate - baseline,
        add_alphas=alphas,
        add_scores=add_results,
        add_deltas={a: add_results[a] - baseline for a in add_results},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--tag", default=None)
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=50,
                   help="smaller than the main protocol (50 vs 100) "
                   "to keep wall-clock manageable across 6 alpha values")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    p.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    tag = args.tag or args.model.replace("/", "_").replace("-", "_").lower()
    out_path = Path(args.output_dir) / f"additive_caa_{tag}.json"

    runner = LlamaModelRunner(model_id=args.model)
    print(f"[additive] model={args.model}  layer={args.layer}  "
          f"tasks={args.tasks}  alphas={args.alphas}", flush=True)

    task_classes = [c for c in ALL_TASKS if c.name in set(args.tasks)]
    results = []
    for cls in task_classes:
        task = cls()
        print(f"\n----- {task.name} ({task.hypothesized_kind}) -----",
              flush=True)
        r = evaluate_additive(task, runner, layer=args.layer,
                              n_pairs=args.n_pairs, n_eval=args.n_eval,
                              alphas=args.alphas)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(dict(model=args.model, layer=args.layer,
                           tasks=results), f, indent=2)

    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
