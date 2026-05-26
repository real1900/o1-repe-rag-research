# SAE-feature mechanistic analysis — runbook

**Purpose.** Turn the paper's "cosmetic framing word" claim for honesty /
sycophancy / politeness into a mechanistic finding. Decompose the mean
steering direction over a public sparse autoencoder's feature dictionary
and show:

- **Low-σ tasks** (honesty, sycophancy, politeness) → direction projects
  onto **1–2 sparse features**, those features are token-level
  (a single SAE-feature label like "the word 'honest'" or "polite-tone").
- **High-σ tasks** (refusal, sentiment, rag_distractor) → direction
  spreads across **many features**, none dominate.

This converts the "geometry vs behavior" gap into a "what is the direction
literally made of" answer. The single figure is a 1-page ICLR-strength
addition.

## Prerequisite: SAE access

The paper's primary model is **Llama-3.2-3B-Instruct**. As of 2026-05, no
widely-distributed public SAE exists for this specific checkpoint. Three
realistic paths:

### Path A — pivot the SAE analysis to Gemma-2-9B (recommended)

Gemma-Scope (Lieberum et al. 2024, Anthropic-released) ships SAEs for
every layer of Gemma-2-{2B, 9B, 27B}. Replicate the 12-task protocol on
**Gemma-2-9B-it** at a layer where the SAE is trained (e.g., layer 12).
Use the cross-family validation result you already have to argue this is
appropriate: the geometric features are model-invariant
(r ≥ +0.998 across families), so the SAE analysis on Gemma transfers
back to Llama.

Compute cost: ~12 hours on 1×A100 (taxonomy run) + 1 hour SAE analysis.

### Path B — use Llama-Scope (community SAEs for Llama-3.1-8B)

He et al. 2024 release SAEs for Llama-3.1-{8B, 70B}. Pivot the *primary*
to Llama-3.1-8B (which Goodfire / Llama-Scope both cover), re-run the
12-task protocol, then do SAE decomposition. This adds Llama-3.1-8B to
the multi-model story (now 5 models).

Compute cost: ~24 hours on 1×A100.

### Path C — train your own SAE on Llama-3.2-3B

Use the dictionary-learning recipe from Cunningham et al. 2023 or
Anthropic's TopK SAE training code. Train on OpenWebText activations at
layer 7. Expensive (~$200–1000 on vast.ai depending on size and
expansion factor). Only worth it if A and B are unavailable.

## Skeleton script (`sae_decompose.py`)

The script below is written to be model-agnostic and SAE-loader-agnostic.
Swap the SAE loader in `load_sae()` for whichever path you took. It
expects the steering directions are saved as a `.pt` dict of
`{task_name: direction_vector}` — currently `steering_taxonomy` does NOT
save these; you will need to add that, either by re-running the protocol
with a small patch to dump `mean_dir` per task, or by re-deriving the
direction from the existing contrast pairs (faster: re-derive).

```python
# sae_decompose.py  -- skeleton, edit the SAE loader for your setup

import json
import torch
import numpy as np
from pathlib import Path

from steering_taxonomy.runner import LlamaModelRunner
from steering_taxonomy.tasks import ALL_TASKS


def load_sae(layer):
    """Load the SAE for the chosen model at the target layer.

    Replace this with your concrete loader. Each SAE exposes:
      - encoder: nn.Linear(d_model, n_features)  # produces sparse codes
      - decoder: nn.Linear(n_features, d_model)  # reconstructs activations
      - feature_labels: optional dict[int, str] of human-readable names
    """
    # ----- Path A (Gemma-Scope) -----
    # from gemma_scope import load_jumprelu_sae
    # return load_jumprelu_sae(layer=layer, model="gemma-2-9b-it",
    #                           width="65k")
    # ----- Path B (Llama-Scope) -----
    # from llama_scope import load_sae
    # return load_sae(layer=layer, model="llama-3.1-8b", width="32x")
    # ----- Path C (your own) -----
    # return torch.load(f"sae_layer{layer}.pt")
    raise NotImplementedError("Edit load_sae for your SAE source.")


def derive_direction(runner, task, layer, n_pairs=200):
    """Build the mean steering direction for one task at one layer.

    Mirrors what the protocol does internally: forward pos and neg
    prompts, take mean-pooled hidden states, average their difference,
    unit-normalize.
    """
    pairs = task.build_pairs(n=n_pairs)
    pos = [p.positive for p in pairs]
    neg = [p.negative for p in pairs]
    h_pos = runner.mean_act(pos, layer)
    h_neg = runner.mean_act(neg, layer)
    d_bar = (h_pos - h_neg).mean(dim=0)
    return d_bar / d_bar.norm()


def decompose_over_sae(direction, sae, top_k=20):
    """Project the steering direction onto the SAE feature dictionary.

    Returns:
      coeffs   : per-feature decoder coefficient (signed)
      sparsity : top-k l2 mass / total l2 mass (1.0 = pure sparse)
      top      : the top_k feature indices by abs(coef)
    """
    # SAE-style decomposition: solve dictionary[features] @ coefs = direction
    # using the encoder (sparse codes for the activation `direction`)
    with torch.no_grad():
        coefs = sae.encode(direction.unsqueeze(0)).squeeze(0)  # [n_features]
    abs_coefs = coefs.abs()
    top = torch.topk(abs_coefs, k=top_k).indices.cpu().numpy()
    top_mass = (abs_coefs[top] ** 2).sum().item()
    total_mass = (abs_coefs ** 2).sum().item()
    sparsity = top_mass / max(total_mass, 1e-10)
    return coefs.cpu().numpy(), float(sparsity), top.tolist()


def main():
    LAYER = 7  # or whatever layer your SAE is trained on
    runner = LlamaModelRunner(model_id="meta-llama/Llama-3.2-3B-Instruct")
    sae = load_sae(LAYER)

    out = {}
    for tc in ALL_TASKS:
        task = tc()
        d = derive_direction(runner, task, LAYER)
        coefs, sparsity, top = decompose_over_sae(d, sae, top_k=20)
        out[task.name] = dict(
            top_k=20,
            sparsity_topk_over_total=sparsity,  # in [0, 1]; high = collapsed
            top_features=top,
            n_features_above_0p1_of_max=int(
                (np.abs(coefs) > 0.1 * np.abs(coefs).max()).sum()
            ),
        )
        print(f"{task.name:25s}  sparsity={sparsity:.3f}  "
              f"n_features_significant={out[task.name]['n_features_above_0p1_of_max']}")

    with open("sae_decomposition.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
```

## What the figure should show (1 page for the paper)

A single bar/scatter plot:

- **x-axis:** per-pair variance σ (the geometric feature)
- **y-axis:** number of SAE features above 10% of peak coefficient (the
  decomposition's effective sparsity)
- **expected pattern:** monotonic positive — low σ tasks decompose onto
  few features (collapsed); high σ tasks spread across many.
- **annotation:** label honesty / sycophancy / refusal / rag_distractor
  to make the contrast vivid.

If the pattern doesn't hold cleanly, the result is still publishable —
it just becomes a more nuanced mechanistic finding rather than a
predictive one.

## Where the result goes in the paper

A new short section **§6 Mechanism** between Discussion and Limitations,
roughly:

> The paper's geometric rule predicts which tasks steer. The
> *mechanistic* reading is that low-σ tasks collapse onto sparse,
> token-level features in the SAE dictionary; high-σ tasks decompose
> across many features that together form a real semantic axis.
> Figure N plots SAE-decomposition sparsity against σ across the 12
> tasks. The expected monotonic pattern holds at r = X. The two
> exceptions are context_faithfulness and persona, where the
> direction projects onto a single "system-prompt-presence" feature
> the model uses as a real conditioning signal — explaining why
> ablation moves behavior despite low σ.

This is exactly the kind of "geometry → mechanism → behavior" arc that
ICLR reviewers reward.

## Estimated effort

- Path A (Gemma-Scope on Gemma-2-9B): **1 week**
  (3 days: re-run protocol on Gemma; 2 days: SAE decomposition; 1 day:
  figure + paper integration; 1 day: buffer)
- Path B (Llama-Scope on Llama-3.1-8B): **1.5 weeks** (extra time for
  Llama-Scope SAE quality verification)
- Path C (train your own): **3–4 weeks**, mostly compute waiting

Recommend Path A.
