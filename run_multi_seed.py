"""Multi-seed runner: re-runs the steering protocol with multiple seeds
and reports mean +- std for the empirical effect columns.

Addresses the reviewer concern: the main results use a single fixed seed.
This script reruns the headline tasks under multiple seeds and produces a
multi-seed report so we can report mean +- std for the Delta values.

Usage on M4 (work dir ~/selfrag_run/):

    cd ~/selfrag_run
    git pull
    source ~/selfrag_env/bin/activate
    python run_multi_seed.py \\
        --tasks refusal sentiment rag_distractor honesty hallucination_grounding \\
        --seeds 2026 1729 4096 \\
        --output-dir taxonomy_multiseed

A multi-seed run on the 5 headline tasks takes ~3 single-seed runs of those
5 tasks. Budget at the same cost-per-task as the main run; total ~4--5 hours
on M4 at N=200 pairs, M=100 eval.

Output: taxonomy_multiseed/<task>_multiseed.json with per-seed deltas plus
mean +- std summary, ready for the paper to cite.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.protocol import evaluate_task
from steering_taxonomy.tasks import ALL_TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*",
                   help="Tasks to run multi-seed. Default: refusal sentiment "
                        "rag_distractor honesty hallucination_grounding "
                        "context_faithfulness persona.")
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 1729, 4096],
                   help="Random seeds to average over.")
    p.add_argument("--output-dir", default="taxonomy_multiseed")
    args = p.parse_args()

    default_tasks = ["refusal", "sentiment", "rag_distractor",
                     "honesty", "hallucination_grounding",
                     "context_faithfulness", "persona"]
    task_names = args.tasks if args.tasks else default_tasks

    task_classes = [t for t in ALL_TASKS if t.name in set(task_names)]
    if not task_classes:
        raise SystemExit(f"No matching tasks. Available: "
                         f"{[t.name for t in ALL_TASKS]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    runner = LlamaModelRunner(model_id=args.model)

    print(f"Multi-seed run: model={args.model}, layer={args.layer}, "
          f"seeds={args.seeds}")
    print(f"Tasks: {[tc.name for tc in task_classes]}")
    t0 = time.time()

    for tc in task_classes:
        task_reports = []
        for seed in args.seeds:
            print(f"\n----- {tc.name}  seed={seed} -----")
            r = evaluate_task(
                tc(), runner,
                target_layer=args.layer,
                n_pairs=args.n_pairs,
                n_eval=args.n_eval,
                n_random=args.n_random,
                seed=seed,
            )
            task_reports.append(dict(seed=seed, **r.to_dict()))
            print(f"  delta_base={r.delta_vs_baseline:+.3f}  "
                  f"delta_rand={r.delta_vs_random:+.3f}  "
                  f"cos={r.split_half_cos_mean:.3f}  var={r.per_pair_variance:.3f}")

        def agg(field):
            vals = [r[field] for r in task_reports]
            return dict(
                mean=statistics.mean(vals),
                std=statistics.stdev(vals) if len(vals) > 1 else 0.0,
                values=vals,
            )

        summary = dict(
            task_name=tc.name,
            model=args.model,
            layer=args.layer,
            n_pairs=args.n_pairs,
            n_eval=args.n_eval,
            n_seeds=len(args.seeds),
            seeds=args.seeds,
            delta_vs_baseline=agg("delta_vs_baseline"),
            delta_vs_random=agg("delta_vs_random"),
            split_half_cos_mean=agg("split_half_cos_mean"),
            per_pair_variance=agg("per_pair_variance"),
            baseline_metric=agg("baseline_metric"),
            steered_metric=agg("steered_metric"),
            random_metric_mean=agg("random_metric_mean"),
            per_seed_reports=task_reports,
        )

        out_path = out_dir / f"{tc.name}_multiseed.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        d_rand = summary["delta_vs_random"]
        d_base = summary["delta_vs_baseline"]
        print(f"\n  SUMMARY {tc.name}:")
        print(f"    delta_vs_baseline = {d_base['mean']:+.3f} +- "
              f"{d_base['std']:.3f}")
        print(f"    delta_vs_random   = {d_rand['mean']:+.3f} +- "
              f"{d_rand['std']:.3f}")
        print(f"    -> {out_path}")

    elapsed = time.time() - t0
    print(f"\nAll done. {len(task_classes) * len(args.seeds)} "
          f"task-seed runs in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
