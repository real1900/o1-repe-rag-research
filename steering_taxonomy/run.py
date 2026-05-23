"""CLI: run the steering-taxonomy protocol across tasks.

Each invocation:
  1. Loads a model via `LlamaModelRunner`.
  2. For each requested task: builds the CAA direction, computes the geometric
     characterization, runs steering + random-direction controls on held-out
     eval, scores with the task's metric.
  3. Writes a per-task `TaskReport` JSON to `--output-dir`.

Usage:
    python -m steering_taxonomy.run                    # all 12 tasks
    python -m steering_taxonomy.run refusal honesty    # specific tasks
    python -m steering_taxonomy.run --model meta-llama/Llama-3.2-3B-Instruct \\
                                     --layer 7 --n-eval 200

Default model: Llama-3.2-3B-Instruct (matches the rest of the project).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.protocol import evaluate_task
from steering_taxonomy.tasks import ALL_TASKS


def main():
    parser = argparse.ArgumentParser(description="Run the steering-taxonomy protocol.")
    parser.add_argument("tasks", nargs="*",
                        help="Task names to run; empty = all 12.")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--layer", type=int, default=7,
                        help="Target layer for steering direction extraction + injection.")
    parser.add_argument("--n-pairs", type=int, default=400,
                        help="Number of contrastive pairs for direction construction.")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of held-out examples for scoring.")
    parser.add_argument("--n-random", type=int, default=5,
                        help="Number of random-direction control vectors.")
    parser.add_argument("--output-dir", default="taxonomy_reports")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Filter tasks by name if any were specified
    task_classes = ALL_TASKS
    if args.tasks:
        wanted = set(args.tasks)
        task_classes = [t for t in ALL_TASKS if t.name in wanted]
        missing = wanted - {t.name for t in task_classes}
        if missing:
            print(f"WARNING: unknown task names skipped: {sorted(missing)}")

    if not task_classes:
        raise SystemExit("No tasks to run.")

    print(f"Running {len(task_classes)} task(s) on {args.model} "
          f"at layer {args.layer}, n_pairs={args.n_pairs}, n_eval={args.n_eval}")
    runner = LlamaModelRunner(model_id=args.model)

    for task_class in task_classes:
        task = task_class()
        print(f"\n----- {task.name}  ({task.hypothesized_kind}) -----")
        report = evaluate_task(
            task, runner,
            target_layer=args.layer,
            n_pairs=args.n_pairs,
            n_eval=args.n_eval,
            n_random=args.n_random,
            seed=args.seed,
        )
        out_path = out_dir / f"{task.name}.json"
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  EM: baseline={report.baseline_metric:.3f}  "
              f"steered={report.steered_metric:.3f}  "
              f"random={report.random_metric_mean:.3f}±{report.random_metric_std:.3f}")
        print(f"  geom: split-half-cos={report.split_half_cos_mean:.3f}"
              f"±{report.split_half_cos_std:.3f}  "
              f"per-pair-var={report.per_pair_variance:.3f}")
        print(f"  delta_vs_baseline={report.delta_vs_baseline:+.3f}  "
              f"delta_vs_random={report.delta_vs_random:+.3f}")
        print(f"  -> {out_path}  ({report.seconds:.0f}s)")

    print(f"\nDone. {len(task_classes)} report(s) in {out_dir}/")


if __name__ == "__main__":
    main()
