"""Geometric-only layer sweep: compute (cos, var) at every layer for every task.

The 12-task taxonomy was run at a single layer per model (Llama layer 7,
Qwen layer 9). This sweeps cos + var across all layers for all tasks, which
is cheap (forward passes only -- no generation). The result is a 2-D table
(task x layer) of geometric signatures; we can then check whether the
predictive rule's classification is stable across layers.

We do NOT run the empirical (steered / random) sweep here -- that would cost
~25 min per (task, layer) cell. Instead this script feeds a much cheaper
geometric layer profile per task; we anchor empirically at a few extra layers
in a separate run.

Output: layer_sweep_<model_tag>.json with shape
    {
      "model_id": "meta-llama/Llama-3.2-3B-Instruct",
      "n_layers": 28,
      "tasks": {
        "refusal":  {"0": {"cos": ..., "var": ...}, "1": ..., ..., "27": ...},
        "honesty":  {...},
        ...
      }
    }

Usage:
    python layer_sweep.py
    python layer_sweep.py --model Qwen/Qwen2.5-3B-Instruct --tag qwen3b
    python layer_sweep.py --tasks refusal honesty rag_distractor context_faithfulness
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.protocol import split_half_cosine, per_pair_variance
from steering_taxonomy.tasks import ALL_TASKS


def sweep_one_task(runner, task, n_pairs, layers):
    """Compute (cos, var) at every layer in `layers` for one task."""
    pairs = task.build_pairs(n=n_pairs)
    pos_texts = [p.positive for p in pairs]
    neg_texts = [p.negative for p in pairs]

    out = {}
    for layer in layers:
        t0 = time.time()
        pos = runner.mean_act(pos_texts, layer=layer)
        neg = runner.mean_act(neg_texts, layer=layer)
        per_pair = pos - neg
        cos_mean, cos_std = split_half_cosine(per_pair)
        var = per_pair_variance(per_pair)
        out[str(layer)] = dict(cos=cos_mean, cos_std=cos_std, var=var,
                               seconds=time.time() - t0)
        print(f"    layer={layer:2d}: cos={cos_mean:.3f}+-{cos_std:.3f}  "
              f"var={var:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--tag", default=None,
                   help="output filename tag; default = model basename")
    p.add_argument("--n-pairs", type=int, default=200,
                   help="contrastive pairs per task (smaller is fine -- "
                   "we only compute geometric features)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="task names; default = all 12")
    p.add_argument("--every-k-layers", type=int, default=1,
                   help="sample every k-th layer (1 = all layers)")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    tag = args.tag or args.model.replace("/", "_").replace("-", "_").lower()
    out_path = Path(args.output_dir) / f"layer_sweep_{tag}.json"

    runner = LlamaModelRunner(model_id=args.model)
    n_layers = len(runner.layers)
    layers = list(range(0, n_layers, args.every_k_layers))
    print(f"[sweep] model={args.model}  n_layers={n_layers}  "
          f"sweeping layers={layers}")

    if args.tasks:
        wanted = set(args.tasks)
        task_classes = [c for c in ALL_TASKS if c.name in wanted]
    else:
        task_classes = ALL_TASKS

    out = dict(model_id=args.model, n_layers=n_layers,
               layers=layers, n_pairs=args.n_pairs, tasks={})
    for cls in task_classes:
        task = cls()
        print(f"\n----- {task.name} -----", flush=True)
        out["tasks"][task.name] = sweep_one_task(runner, task,
                                                 n_pairs=args.n_pairs,
                                                 layers=layers)
        # Save after each task in case of crash
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
