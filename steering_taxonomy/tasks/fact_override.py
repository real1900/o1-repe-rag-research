"""Fact override -- steering the model to state a per-example counterfactual.

Hypothesized content-specific: every example specifies a different fact, so
the contrastive direction depends on the per-example target and should vary
widely across pairs (no single direction).

Distinction from `context_faithfulness` (borderline): context_faithfulness
is a *global disposition* to trust context over parametric memory -- one
stable axis across all inputs. Fact override targets a SPECIFIC counterfactual
per example, so the steering target itself varies by content.

Contrastive setup:
  positive: a question + its specific COUNTERFACTUAL answer
  negative: the same question + its specific FACTUAL answer
Direction:
  mean(counterfactual_acts) - mean(factual_acts) at a target layer.
Eval:
  Held-out questions; score = does the model produce the counterfactual
  answer when steered (1.0) vs the factual one (0.0).
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


class FactOverrideTask(SteeringTask):
    name = "fact_override"
    description = "Inject per-example counterfactual answers (content-specific by design)."
    hypothesized_kind = "content_specific"

    def __init__(self, examples_path: str | Path | None = None, eval_frac: float = 0.25):
        self.examples_path = Path(examples_path) if examples_path else None
        self.eval_frac = eval_frac

    def build_pairs(self, n=None):
        """Each pair: (Q + counterfactual answer, Q + factual answer)."""
        examples = self._load()
        n_train = int(len(examples) * (1 - self.eval_frac))
        examples = examples[:n_train]
        if n is not None:
            examples = examples[:n]
        return [
            ContrastivePair(
                positive=f"Q: {ex['question']}\nA: {ex.get('counterfactual_answer', '?')}",
                negative=f"Q: {ex['question']}\nA: {ex.get('factual_answer', '?')}",
                metadata={"task": self.name, "index": i},
            )
            for i, ex in enumerate(examples)
        ]

    def build_eval(self, n=None):
        examples = self._load()
        start = int(len(examples) * (1 - self.eval_frac))
        held_out = examples[start:]
        if n is not None:
            held_out = held_out[:n]
        return [
            EvalExample(
                prompt=f"Q: {ex['question']}\nA:",
                target={"counterfactual": ex.get("counterfactual_answer"),
                        "factual": ex.get("factual_answer")},
                metadata={"task": self.name},
            )
            for ex in held_out
        ]

    def score_completion(self, completion, example):
        """1.0 if completion contains the counterfactual; 0.0 if it contains the factual."""
        c = completion.lower()
        target = example.target if isinstance(example.target, dict) else {}
        counter = (target.get("counterfactual") or "").lower()
        factual = (target.get("factual") or "").lower()
        if counter and counter in c:
            return 1.0
        if factual and factual in c:
            return 0.0
        return 0.5

    def _load(self):
        """Load (question, counterfactual_answer, factual_answer) triples.

        Uses NQ-Swap (Longpre et al.) -- a Natural Questions variant where
        each item has the original parametric answer and a substituted
        counterfactual answer. Each example has a DIFFERENT counterfactual
        fact -- the content-specific signature.
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        try:
            from datasets import load_dataset
            # NQ-Swap exposes only a `dev` split (no `validation` alias).
            ds = load_dataset("pminervini/NQ-Swap", split="dev")
            def _first(v):
                # NQ-Swap stores answers as lists; collapse to the first string.
                if isinstance(v, list):
                    return v[0].strip() if v else ""
                return (v or "").strip()
            return [
                {"question": row["question"],
                 "counterfactual_answer": _first(row.get("sub_answer")
                                                  or row.get("substituted_answer")),
                 "factual_answer": _first(row.get("org_answer")
                                          or row.get("original_answer"))}
                for row in ds
            ]
        except Exception as e:
            print(f"[fact_override] NQ-Swap unavailable ({e}); using placeholders")
            return [
                {"question": f"<fact-override-q-{i}>",
                 "counterfactual_answer": f"<counterfactual-{i}>",
                 "factual_answer": f"<factual-{i}>"}
                for i in range(100)
            ]
