"""Politeness -- steering toward polite vs rude register.

Borderline: register/style is *plausibly* behavioral (one polite-vs-rude
axis), but politeness can interact with content (some content is more
sensitive to register). Predict mostly-behavioral (high stability).

Contrastive setup:
  positive: prompt + a polite response
  negative: prompt + a rude/curt response
Direction:
  mean(polite_acts) - mean(rude_acts) at a target layer.
Eval:
  Held-out prompts; score = politeness markers in the generated continuation.

Hypothesized as borderline -- expected to lean behavioral.
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


POLITE_MARKERS = ["please", "thank you", "appreciate", "kindly", "would you",
                  "could you", "i'd be happy", "certainly", "of course"]
RUDE_MARKERS = ["whatever", "shut up", "deal with it", "no way", "stupid",
                "idiot", "ridiculous", "ugh"]


class PolitenessTask(SteeringTask):
    name = "politeness"
    description = "Polite vs rude register (borderline)."
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
                positive=f"{ex['prompt']} {ex.get('polite', '')}",
                negative=f"{ex['prompt']} {ex.get('rude', '')}",
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
            EvalExample(prompt=ex["prompt"], target="polite",
                        metadata={"task": self.name}) for ex in held_out
        ]

    def score_completion(self, completion, example):
        """Politeness ratio = polite markers / (polite + rude); 0.5 if neither found."""
        c = completion.lower()
        pol = sum(1 for m in POLITE_MARKERS if m in c)
        rud = sum(1 for m in RUDE_MARKERS if m in c)
        if pol + rud == 0:
            return 0.5
        return float(pol) / (pol + rud)

    def _load(self):
        """Load (prompt, polite, rude) triples.

        TODO(next): source / curate a politeness corpus or generate pairs via
        an LLM with explicit register prompts (e.g., GoEmotions has politeness
        signals; Stanford Politeness Corpus is the classic source).
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        return [
            {"prompt": f"<politeness-prompt-{i}>",
             "polite": "Thank you, I appreciate your time.",
             "rude": "Whatever, deal with it."}
            for i in range(100)
        ]
