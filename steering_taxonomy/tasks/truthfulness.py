"""Truthfulness -- steering toward truthful answers on TruthfulQA.

Reference: Li et al., "Inference-Time Intervention: Eliciting Truthful Answers
from a Language Model" (ITI, NeurIPS 2023). The canonical truthfulness steering
setup. Uses TruthfulQA (Lin et al., 2022) -- multiple-choice questions where
common misconceptions are the wrong answers.

Contrastive setup:
  positive: question + a TRUTHFUL ("best_answer") answer
  negative: question + an INCORRECT (common-misconception) answer
Direction:
  mean(truthful_acts) - mean(incorrect_acts) at a target layer.
Eval:
  Held-out TQA questions; score = is the steered model's answer in the
  truthful answer set, the incorrect answer set, or off-topic?
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


class TruthfulnessTask(SteeringTask):
    name = "truthfulness"
    description = "Steering truthful answers on TruthfulQA (Li et al., 2023)."
    hypothesized_kind = "behavioral"

    def __init__(self, tqa_path: str | Path | None = None, eval_frac: float = 0.25):
        self.tqa_path = Path(tqa_path) if tqa_path else None
        self.eval_frac = eval_frac

    def build_pairs(self, n=None):
        """Each pair: (Q + truthful answer, Q + incorrect answer)."""
        examples = self._load()
        n_train = int(len(examples) * (1 - self.eval_frac))
        examples = examples[:n_train]
        if n is not None:
            examples = examples[:n]
        return [
            ContrastivePair(
                positive=f"Q: {ex['question']}\nA: {ex.get('best_answer', '?')}",
                negative=f"Q: {ex['question']}\nA: {(ex.get('incorrect_answers') or ['?'])[0]}",
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
                target={
                    "correct": ex.get("correct_answers") or [ex.get("best_answer", "")],
                    "incorrect": ex.get("incorrect_answers") or [],
                },
                metadata={"task": self.name},
            )
            for ex in held_out
        ]

    def score_completion(self, completion, example):
        """1.0 if completion matches any correct answer; 0.0 if matches any incorrect."""
        c = completion.lower()
        target = example.target if isinstance(example.target, dict) else {}
        correct = [a.lower() for a in target.get("correct", []) if a]
        incorrect = [a.lower() for a in target.get("incorrect", []) if a]
        if any(ans and ans in c for ans in correct):
            return 1.0
        if any(ans and ans in c for ans in incorrect):
            return 0.0
        return 0.5  # off-topic / unscoreable

    def _load(self):
        """Load TruthfulQA examples (generation split via HF `datasets`)."""
        if self.tqa_path and self.tqa_path.exists():
            with open(self.tqa_path) as f:
                return json.load(f)
        try:
            from datasets import load_dataset
            ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
            return [
                {"question": row["question"],
                 "best_answer": row.get("best_answer", ""),
                 "correct_answers": row.get("correct_answers") or [],
                 "incorrect_answers": row.get("incorrect_answers") or []}
                for row in ds
            ]
        except Exception as e:
            print(f"[truthfulness] TruthfulQA unavailable ({e}); using placeholders")
            return [
                {"question": f"<truthfulness-placeholder-q-{i}>",
                 "best_answer": "yes",
                 "correct_answers": ["yes", "indeed"],
                 "incorrect_answers": ["no", "false"]}
                for i in range(100)
            ]
