"""SAE decomposition of steering directions on Gemma-2-9B-it via Gemma-Scope.

Implements the mechanism analysis described in `sae_analysis_PLAN.md`,
Path A. For each task in the 12-task corpus:
  1. Build the mean steering direction `d_bar` at the SAE's target layer.
  2. Encode `d_bar` through the Gemma-Scope JumpReLU SAE encoder.
  3. Measure how concentrated the resulting feature activation is.

Hypothesis under test (paper claim):
  - LOW per-pair variance (collapsed/cosmetic contrasts: honesty,
    sycophancy, politeness, topic_suppression) -> direction decomposes
    onto FEW SAE features (sparse, token-level).
  - HIGH per-pair variance (genuine axis: refusal, sentiment,
    rag_distractor) -> direction decomposes onto MANY SAE features.

Output: sae_decomposition.json -- per-task feature counts, top features,
sparsity. A reviewer can verify the mechanism without re-running.

Usage on M4 (after Gemma protocol on M4 produces a directions cache):

    cd ~/selfrag_run
    source ~/selfrag_env/bin/activate
    python sae_decompose.py --layer 9 --out-json sae_decomposition.json

This script DERIVES the directions itself by re-running the contrast
forward passes through Gemma-2-9B-it; it does not depend on the
taxonomy_reports/ output files (which save only summary stats).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.tasks import ALL_TASKS


# ---------------------------------------------------------------------------
# Gemma-Scope JumpReLU SAE loader
# ---------------------------------------------------------------------------

class JumpReLUSAE:
    """Minimal Gemma-Scope JumpReLU SAE.

    The SAE has encoder weights W_enc (d_model, n_features), decoder
    weights W_dec (n_features, d_model), biases, and a per-feature
    JumpReLU threshold. Forward pass:
        pre = (x - b_dec) @ W_enc + b_enc
        z   = pre * (pre > threshold)            # JumpReLU
        recon = z @ W_dec + b_dec

    For our analysis we only need the encoder (project the steering
    direction into feature space).
    """

    def __init__(self, params_npz_path, device="cpu"):
        params = np.load(params_npz_path)
        self.W_enc = torch.from_numpy(params["W_enc"]).to(device)
        self.W_dec = torch.from_numpy(params["W_dec"]).to(device)
        self.b_enc = torch.from_numpy(params["b_enc"]).to(device)
        self.b_dec = torch.from_numpy(params["b_dec"]).to(device)
        self.threshold = torch.from_numpy(params["threshold"]).to(device)
        self.n_features = self.W_enc.shape[1]
        self.d_model = self.W_enc.shape[0]
        self.device = device
        print(f"[sae] loaded: d_model={self.d_model}, "
              f"n_features={self.n_features}")

    @torch.no_grad()
    def encode(self, x):
        """JumpReLU encode: returns feature activations [..., n_features]."""
        x = x.to(self.device).float()
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        z = pre * (pre > self.threshold).float()
        return z


# ---------------------------------------------------------------------------
# Direction derivation + decomposition
# ---------------------------------------------------------------------------

def derive_direction(runner, task, layer, n_pairs=200):
    """Build the unit-norm mean steering direction at the target layer."""
    pairs = task.build_pairs(n=n_pairs)
    if len(pairs) < 20:
        return None
    pos = [p.positive for p in pairs]
    neg = [p.negative for p in pairs]
    h_pos = runner.mean_act(pos, layer)
    h_neg = runner.mean_act(neg, layer)
    d_bar = (h_pos - h_neg).mean(dim=0)
    return d_bar / d_bar.norm()


def decompose(direction, sae):
    """Decompose direction over SAE features; return summary stats."""
    # JumpReLU only activates for ALIGNED inputs (positive pre-activations).
    # The steering direction may have either sign; we look at both directions
    # and report the larger response.
    z_pos = sae.encode(direction)
    z_neg = sae.encode(-direction)
    z_pos_abs = z_pos.abs().cpu().numpy()
    z_neg_abs = z_neg.abs().cpu().numpy()
    # Take the side with more total activation (the model presumably
    # encodes the contrast in one consistent sign).
    if z_pos_abs.sum() >= z_neg_abs.sum():
        z = z_pos_abs
        sign = "+"
    else:
        z = z_neg_abs
        sign = "-"

    # How concentrated is this decomposition? Use:
    #   - count of features above 10% of peak
    #   - top-20 features and their decoder magnitudes
    #   - Gini coefficient as a continuous concentration measure
    peak = z.max() if z.max() > 0 else 1e-10
    n_features_significant = int((z > 0.1 * peak).sum())
    n_features_active = int((z > 0).sum())
    top20_idx = np.argsort(-z)[:20]
    top20_mass = float(z[top20_idx].sum())
    total_mass = float(z.sum())
    top20_fraction = top20_mass / max(total_mass, 1e-10)

    # Gini: 1 - sum(squared shares). 0 = uniform, ~1 = single feature dominates.
    shares = z / max(z.sum(), 1e-10)
    gini = float((shares ** 2).sum())

    return dict(
        sign=sign,
        n_features_active=n_features_active,
        n_features_significant=n_features_significant,
        top20_features=[int(i) for i in top20_idx],
        top20_activations=[float(z[i]) for i in top20_idx],
        top20_fraction_of_total=top20_fraction,
        gini=gini,
        peak_activation=float(peak),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--layer", type=int, default=9,
                   help="Target layer; Gemma-Scope 9b-it has SAEs at "
                        "layers 9, 20, 31.")
    p.add_argument("--sae-repo", default="google/gemma-scope-9b-it-res")
    p.add_argument("--sae-path",
                   default="layer_9/width_16k/average_l0_47/params.npz",
                   help="Path within --sae-repo to the SAE's params.npz")
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--out-json", default="sae_decomposition.json")
    args = p.parse_args()

    print(f"[main] Gemma-Scope SAE: {args.sae_repo} :: {args.sae_path}")
    sae_path = hf_hub_download(repo_id=args.sae_repo, filename=args.sae_path)
    print(f"[main] SAE local path: {sae_path}")

    # Load SAE on CPU (small, no MPS pressure) before loading the big model
    sae = JumpReLUSAE(sae_path, device="cpu")

    print(f"[main] loading model {args.model} ...")
    runner = LlamaModelRunner(model_id=args.model)
    print(f"[main] runner ready: hidden={runner.d_model}, "
          f"layers={len(runner.layers)}")
    assert runner.d_model == sae.d_model, \
        f"Mismatch: model hidden {runner.d_model} != SAE d_model {sae.d_model}"

    results = {}
    t0 = time.time()
    for tc in ALL_TASKS:
        task = tc()
        print(f"\n----- {task.name} -----")
        t1 = time.time()
        d = derive_direction(runner, task, args.layer, n_pairs=args.n_pairs)
        if d is None:
            print("  too few pairs, skipping")
            continue
        decomp = decompose(d.cpu(), sae)
        decomp["task_name"] = task.name
        decomp["task_kind"] = task.hypothesized_kind
        decomp["target_layer"] = args.layer
        decomp["seconds"] = time.time() - t1
        results[task.name] = decomp
        print(f"  sign: {decomp['sign']}")
        print(f"  n_features_significant (above 0.1*peak): "
              f"{decomp['n_features_significant']}")
        print(f"  top-20 fraction of total activation: "
              f"{decomp['top20_fraction_of_total']:.3f}")
        print(f"  Gini (concentration): {decomp['gini']:.3f}")
        print(f"  ({decomp['seconds']:.0f}s)")

    summary = dict(
        model=args.model,
        layer=args.layer,
        sae_repo=args.sae_repo,
        sae_path=args.sae_path,
        n_pairs_per_task=args.n_pairs,
        per_task=results,
    )
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_json}")
    print(f"Total time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
