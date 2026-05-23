"""Unified evaluation + geometric characterization protocol.

Given a `SteeringTask` and a `model_runner` (any object exposing `.mean_act()`
and `.generate()`), this:

  1. Builds the CAA-style steering direction from the task's contrastive pairs.
  2. Computes geometric characterizations (split-half cosine, per-pair variance).
  3. Runs the steering intervention + random-direction controls on held-out eval.
  4. Scores with the task's metric.
  5. Returns a uniform `TaskReport` for cross-task comparison.

The model_runner interface (duck-typed -- any object satisfying it):

    class ModelRunner:                   # Protocol
        d_model: int
        def mean_act(self, texts: list[str], layer: int) -> torch.Tensor: ...
            # [len(texts), d_model] float32 tensor of mean-pooled hidden states.
        def generate(self, prompt: str, hook: tuple | None, layer: int,
                     max_new_tokens: int) -> str: ...
            # `hook` is either None (no intervention) or ("ablate", unit_direction).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from steering_taxonomy.base import SteeringTask


# ---------------------------------------------------------------------------
# Geometric characterization (pure tensor analysis -- no model forward passes)
# ---------------------------------------------------------------------------

def split_half_cosine(per_pair_directions: torch.Tensor,
                      n_splits: int = 20, seed: int = 2026) -> tuple[float, float]:
    """Stability score: cosine similarity between two halves of the pair set.

    High (-> 1) means the task has a stable axis: any half of the pairs gives
    essentially the same direction. Low (-> 0 or negative) means there is no
    single direction -- the per-pair directions disagree across the pair set.

    Returns: (mean, std) over `n_splits` random splits.
    """
    n = per_pair_directions.shape[0]
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        half = n // 2
        a = per_pair_directions[perm[:half]].mean(0)
        b = per_pair_directions[perm[half:half * 2]].mean(0)
        cos = (a @ b / (a.norm() * b.norm() + 1e-9)).item()
        cosines.append(cos)
    return float(np.mean(cosines)), float(np.std(cosines))


def per_pair_variance(per_pair_directions: torch.Tensor) -> float:
    """Average angular deviation of per-pair directions from their mean.

    0 = all per-pair directions perfectly aligned with the mean (a stable axis).
    1 = orthogonal on average (no single direction).
    """
    mean = per_pair_directions.mean(0)
    mean_unit = mean / (mean.norm() + 1e-9)
    cos_to_mean = (per_pair_directions @ mean_unit) / (per_pair_directions.norm(dim=1) + 1e-9)
    return float(1.0 - cos_to_mean.mean().item())


# ---------------------------------------------------------------------------
# Result report (uniform across all corpus tasks for cross-task comparison)
# ---------------------------------------------------------------------------

@dataclass
class TaskReport:
    task_name: str
    task_kind: str
    n_pairs: int
    n_eval: int
    target_layer: int
    # Geometric characterization
    split_half_cos_mean: float
    split_half_cos_std: float
    per_pair_variance: float
    # Effect measurements
    baseline_metric: float
    steered_metric: float
    random_metric_mean: float
    random_metric_std: float
    delta_vs_baseline: float
    delta_vs_random: float
    # Bookkeeping
    seconds: float = 0.0
    notes: str = ""

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------------
# End-to-end: run one task
# ---------------------------------------------------------------------------

def evaluate_task(task: SteeringTask, model_runner, target_layer: int = 7,
                  n_pairs: int = 400, n_eval: int = 200, n_random: int = 5,
                  seed: int = 2026) -> TaskReport:
    """Run the full taxonomy protocol on one task. Returns a uniform `TaskReport`."""
    t0 = time.time()
    pairs = task.build_pairs(n=n_pairs)
    eval_examples = task.build_eval(n=n_eval)

    # 1. Extract per-pair activations at the target layer
    pos_acts = model_runner.mean_act([p.positive for p in pairs], layer=target_layer)
    neg_acts = model_runner.mean_act([p.negative for p in pairs], layer=target_layer)
    per_pair_dirs = pos_acts - neg_acts                # [n_pairs, d_model]

    # 2. Geometric characterization
    cos_mean, cos_std = split_half_cosine(per_pair_dirs)
    var = per_pair_variance(per_pair_dirs)

    # 3. Build the mean CAA direction (unit-norm)
    direction = per_pair_dirs.mean(0)
    direction = direction / (direction.norm() + 1e-9)

    # 4. Random-direction controls (n_random unit vectors)
    gen = torch.Generator().manual_seed(seed)
    random_dirs = []
    for _ in range(n_random):
        rd = torch.randn(direction.shape[0], generator=gen)
        random_dirs.append(rd / rd.norm())

    # 5. Measure: baseline (no hook) + steered (CAA-ablate) + random_* (random-ablate)
    baseline = _score_set(model_runner, task, eval_examples,
                          hook=None, layer=target_layer)
    steered = _score_set(model_runner, task, eval_examples,
                         hook=("ablate", direction), layer=target_layer)
    randoms = [
        _score_set(model_runner, task, eval_examples,
                   hook=("ablate", rd), layer=target_layer)
        for rd in random_dirs
    ]

    return TaskReport(
        task_name=task.name, task_kind=task.hypothesized_kind,
        n_pairs=len(pairs), n_eval=len(eval_examples), target_layer=target_layer,
        split_half_cos_mean=cos_mean, split_half_cos_std=cos_std,
        per_pair_variance=var,
        baseline_metric=baseline, steered_metric=steered,
        random_metric_mean=float(np.mean(randoms)),
        random_metric_std=float(np.std(randoms)),
        delta_vs_baseline=steered - baseline,
        delta_vs_random=steered - float(np.mean(randoms)),
        seconds=time.time() - t0,
    )


def _score_set(model_runner, task, examples, hook, layer):
    """Generate completions (optionally with steering hook) and average the task score."""
    scores = []
    for ex in examples:
        completion = model_runner.generate(ex.prompt, hook=hook,
                                           layer=layer, max_new_tokens=64)
        scores.append(task.score_completion(completion, ex))
    return float(np.mean(scores))
