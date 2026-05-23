"""Context faithfulness -- steering the model to trust context over parametric memory.

Reference: ContextFocus (Yang et al., 2026, arXiv:2601.04131). Reported a
large effect (35% -> 71% on ConFiQA's substituted-answer match) with a
CAA-style residual-stream intervention. ContextFocus did NOT report a
random-direction control; the taxonomy protocol adds it.

Contrastive setup (ContextFocus pattern):
  positive: (system instruction + context + question) -- a context-grounded prompt
  negative: (question only) -- bare question, no context
Direction:
  mean(with-context_acts) - mean(no-context_acts) at a target layer.
  This isolates the "use-context" disposition.
Eval:
  Held-out questions where the provided context conflicts with parametric
  memory (counterfactual context); score = does the model follow the context
  (1.0) or stay with its parametric answer (0.0).

Hypothesized as borderline. "Use context" is a global behavioral disposition
(predicted high split-half stability) even though *which* fact the model
trusts varies per example. The contrast (with-context vs no-context) factors
out the per-example content, leaving the disposition.
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


SYSTEM_INSTRUCTION = (
    "Answer the question using ONLY the provided context. If the context "
    "contradicts what you know, follow the context."
)


class ContextFaithfulnessTask(SteeringTask):
    name = "context_faithfulness"
    description = "Trust context over parametric memory (ContextFocus, 2026)."
    hypothesized_kind = "borderline"

    def __init__(self, examples_path: str | Path | None = None, eval_frac: float = 0.25):
        self.examples_path = Path(examples_path) if examples_path else None
        self.eval_frac = eval_frac

    def build_pairs(self, n=None):
        examples = self._load()
        n_train = int(len(examples) * (1 - self.eval_frac))
        examples = examples[:n_train]
        if n is not None:
            examples = examples[:n]
        return [
            ContrastivePair(
                positive=(f"{SYSTEM_INSTRUCTION}\n\nContext: {ex['context']}\n\n"
                          f"Question: {ex['question']}"),
                negative=f"Question: {ex['question']}",
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
                prompt=(f"{SYSTEM_INSTRUCTION}\n\nContext: {ex['context']}\n\n"
                        f"Question: {ex['question']}\nAnswer:"),
                target={"context_answer": ex.get("substituted_answer"),
                        "parametric_answer": ex.get("original_answer")},
                metadata={"task": self.name},
            )
            for ex in held_out
        ]

    def score_completion(self, completion, example):
        """1.0 if completion matches the context-grounded (substituted) answer."""
        c = completion.lower()
        target = example.target if isinstance(example.target, dict) else {}
        ctx_ans = (target.get("context_answer") or "").lower()
        para_ans = (target.get("parametric_answer") or "").lower()
        if ctx_ans and ctx_ans in c:
            return 1.0
        if para_ans and para_ans in c:
            return 0.0
        return 0.5

    def _load(self):
        """Load (question, context, substituted_answer, original_answer) examples.

        TODO(next): wire to ConFiQA (the dataset ContextFocus uses). Each item
        has a question, a counterfactual context that overrides a parametric
        fact, the new (substituted) answer, and the original (parametric) one.
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        return [
            {"question": f"<ctxfaith-q-{i}>",
             "context": f"<counterfactual context for q-{i}>",
             "substituted_answer": "B",
             "original_answer": "A"}
            for i in range(100)
        ]
