# Steering Taxonomy

A controlled study of *when* activation steering works for LLM tasks — and when it doesn't.

## Research question

> *Predict, from the structure of a task, whether activation steering will succeed.*

## Working hypothesis

Steering succeeds iff the task corresponds to a single stable axis in activation
space — high split-half cosine stability between contrastive directions built on
disjoint halves of the pair set, and low per-example variance. It fails when the
target direction depends on per-example content (no single direction exists).

## Layout

- `base.py` — abstract `SteeringTask` interface every task implements.
- `protocol.py` — unified evaluation + geometric characterization pipeline.
- `tasks/` — one module per task in the corpus.

## Corpus (12 tasks)

**Behavioral — expect steering to succeed:**

1. `honesty` — Zou et al. (RepE)
2. `refusal` — Arditi et al.
3. `sycophancy` — Rimsky et al. (CAA)
4. `sentiment`
5. `truthfulness` — Li et al. (ITI on TruthfulQA)

**Content-specific — expect steering to fail:**

6. `rag_distractor` — MACH-1 distractor suppression (existing pilot data)
7. `fact_override` — per-example fact substitution
8. `topic_suppression` — refuse a specific named topic

**Borderline — the discriminating cases:**

9. `context_faithfulness` — ContextFocus (2026)
10. `persona`
11. `politeness`
12. `hallucination_grounding`

## Protocol

For every task, identically:

1. **Build the CAA direction** from contrastive pairs:
   `dir = mean(positive_acts) - mean(negative_acts)` at a target layer, normalized.
2. **Geometric characterization** (pure tensor analysis, no model forward passes):
   - *Split-half cosine* — cosine similarity between directions built on disjoint halves; measures axis stability.
   - *Per-pair variance* — average angular deviation of per-pair directions from the mean.
3. **Apply steering** (ablation — scale-free) and **random-direction controls** on the held-out eval set.
4. **Score** with the task's metric.
5. **Cross-task** — search for predictive features distinguishing success from failure.
