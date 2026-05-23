"""RAG distractor suppression -- steering the model to ignore retrieval distractors.

Reference: MACH-1 pilot (this project, 2026). Already tested with rigorous
random-direction controls; the pilot found a clean null -- properly-built CAA
ablation does not recover the distractor damage (clean 63.6%, distracted 54.2%,
caa_ablate 53.8% / -0.4pp p=0.50, indistinguishable from random_ablate 54.2%).

Contrastive setup:
  positive: a distractor passage (off-topic / irrelevant retrieved content)
  negative: a gold passage (relevant retrieved content)
Direction:
  mean(distractor_acts) - mean(gold_acts) at a target layer.
Eval:
  Held-out HotpotQA questions presented with a mixed gold+distractor retrieved
  set; score = does the model produce the gold answer (substring match).

Hypothesized as content-specific: every distractor is about a different topic,
so there is no single "distractor-ness" direction -- the per-pair contrastive
direction should vary widely across examples (low split-half stability).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


class RagDistractorTask(SteeringTask):
    name = "rag_distractor"
    description = "Suppress retrieval distractors in HotpotQA (MACH-1 pilot data, 2026)."
    hypothesized_kind = "content_specific"

    def __init__(self, data_path: str | Path = "hotpot_rag_5000.json",
                 caa_start: int = 0, caa_end: int = 400,
                 eval_start: int = 3500, eval_end: int = 4000,
                 seed: int = 2026):
        self.data_path = Path(data_path)
        self.caa_start = caa_start
        self.caa_end = caa_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.seed = seed

    def build_pairs(self, n=None):
        rows = self._load()
        caa_rows = rows[self.caa_start:self.caa_end]
        if n is not None:
            caa_rows = caa_rows[:n]
        return [
            ContrastivePair(
                positive=r["distractor_passages"][0],
                negative=r["gold_passages"][0],
                metadata={"task": self.name, "id": r["id"]},
            )
            for r in caa_rows
        ]

    def build_eval(self, n=None):
        rows = self._load()
        eval_rows = rows[self.eval_start:self.eval_end]
        if n is not None:
            eval_rows = eval_rows[:n]
        rng = random.Random(self.seed)
        examples = []
        for r in eval_rows:
            mixed = list(r["gold_passages"]) + list(r["distractor_passages"])
            rng.shuffle(mixed)
            ctx = "\n\n".join(mixed)
            prompt = (
                "You are a precise question-answering assistant. Answer only "
                "with the literal answer span; do not explain.\n\n"
                f"Context: {ctx}\n\nQuestion: {r['question']}"
            )
            examples.append(EvalExample(
                prompt=prompt,
                target=r["answer"].lower(),
                metadata={"task": self.name, "id": r["id"]},
            ))
        return examples

    def score_completion(self, completion, example):
        """1.0 if the gold answer appears as a substring in the completion."""
        target = (example.target if isinstance(example.target, str) else "").lower()
        if not target:
            return 0.0
        return float(target in completion.lower())

    def _load(self):
        """Load the multi-passage HotpotQA dataset (built by prepare_rag_dataset.py)."""
        if not self.data_path.exists():
            # Placeholder so the interface is testable without the real file
            return [
                {"id": f"placeholder-{i}", "question": f"<rag-distractor-q-{i}>",
                 "answer": "placeholder",
                 "gold_passages": ["<gold passage>"],
                 "distractor_passages": ["<distractor passage>"]}
                for i in range(500)
            ]
        with open(self.data_path) as f:
            return json.load(f)
