"""Sentiment -- steering toward positive vs negative continuation sentiment.

A classic, well-established behavioral steering target. The RepE / CAA / ITI
literature includes sentiment as a standard demo of a stable behavioral axis.

Contrastive setup:
  positive: prompt + a positive-sentiment continuation
  negative: prompt + a negative-sentiment continuation
Direction:
  mean(positive_acts) - mean(negative_acts) at a target layer.
Eval:
  Held-out prompts; score = sentiment of the generated continuation
  (1.0 = positive, 0.0 = negative; placeholder uses a lexicon).
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


# Crude lexicon for placeholder scoring -- replace with a real classifier
# (e.g., distilbert-sst2) when wiring real data.
POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "amazing", "love",
    "best", "happy", "fantastic", "delightful", "beautiful", "joy",
    "brilliant", "perfect", "marvelous",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate",
    "sad", "ugly", "disgusting", "boring", "disappointing", "miserable",
    "atrocious", "dreadful", "appalling",
}


class SentimentTask(SteeringTask):
    name = "sentiment"
    description = "Steering positive vs negative continuation sentiment."
    hypothesized_kind = "behavioral"

    def __init__(self, examples_path: str | Path | None = None, eval_frac: float = 0.25):
        self.examples_path = Path(examples_path) if examples_path else None
        self.eval_frac = eval_frac

    def build_pairs(self, n=None):
        """Each pair: (positive completion, negative completion) for the same prompt."""
        examples = self._load()
        n_train = int(len(examples) * (1 - self.eval_frac))
        examples = examples[:n_train]
        if n is not None:
            examples = examples[:n]
        return [
            ContrastivePair(
                positive=f"{ex['prompt']} {ex.get('positive', '')}",
                negative=f"{ex['prompt']} {ex.get('negative', '')}",
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
        # Steering target is "positive" by convention; the protocol applies the
        # direction in both signs to probe the axis.
        return [
            EvalExample(prompt=ex["prompt"], target="positive",
                        metadata={"task": self.name}) for ex in held_out
        ]

    def score_completion(self, completion, example):
        """Lexicon-based positive ratio in [0, 1]; 0.5 if no sentiment words found."""
        words = set(completion.lower().split())
        pos_hit = len(words & POSITIVE_WORDS)
        neg_hit = len(words & NEGATIVE_WORDS)
        if pos_hit + neg_hit == 0:
            return 0.5
        return float(pos_hit) / (pos_hit + neg_hit)

    # Sentiment-eliciting eval prompts (used as the prompt slot for build_eval).
    _EVAL_PROMPTS = [
        "I think this movie was", "The food at the restaurant was",
        "My overall experience was", "I would say the service was",
        "Honestly, the book was", "Looking back, my vacation was",
        "The performance last night was", "The new app feels",
    ]

    def _load(self):
        """Load (prompt, positive, negative) triples from IMDB.

        Pairs positive-labeled and negative-labeled IMDB reviews by index
        (the direction is built from the difference of labeled activations).
        Uses a small set of standard sentiment-eliciting prompts in the
        prompt slot. Replace the lexicon scorer with a sentiment classifier
        when needed for stronger eval.
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        try:
            from datasets import load_dataset
            ds = load_dataset("stanfordnlp/imdb", split="train")
            positives, negatives = [], []
            for row in ds:
                # IMDB labels: 0 = negative, 1 = positive
                if row["label"] == 1 and len(positives) < 500:
                    positives.append(row["text"])
                elif row["label"] == 0 and len(negatives) < 500:
                    negatives.append(row["text"])
                if len(positives) >= 500 and len(negatives) >= 500:
                    break
            k = min(len(positives), len(negatives))
            prompts = self._EVAL_PROMPTS
            return [
                {"prompt": prompts[i % len(prompts)],
                 "positive": positives[i],
                 "negative": negatives[i]}
                for i in range(k)
            ]
        except Exception as e:
            print(f"[sentiment] IMDB unavailable ({e}); using placeholders")
            return [
                {"prompt": f"<sentiment-placeholder-prompt-{i}>",
                 "positive": "It was wonderful and amazing.",
                 "negative": "It was terrible and awful."}
                for i in range(100)
            ]
