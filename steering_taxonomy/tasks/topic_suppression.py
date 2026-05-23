"""Topic suppression -- steering the model to refuse a per-example named topic.

Hypothesized content-specific: every example specifies a different topic, so
the contrastive direction depends on which topic is being suppressed (no
single direction).

Distinction from `refusal` (behavioral): refusal is a *global disposition* to
decline harmful content -- one stable axis across all inputs. Topic
suppression targets a SPECIFIC named topic per example, so the steering
target varies by content.

Contrastive setup:
  positive: prompt asks about topic X + refusal of topic X
  negative: prompt asks about topic X + answer about topic X
Direction:
  mean(refuse-topic-X_acts) - mean(answer-topic-X_acts) over varying X.
Eval:
  Held-out prompts on per-example topics; score = does the model refuse on
  the named topic (1.0) or answer (0.0).
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


REFUSAL_PHRASES = [
    "i cannot", "i can't", "i won't", "i will not",
    "i'm sorry", "i must decline", "i'm unable", "i am unable",
]


class TopicSuppressionTask(SteeringTask):
    name = "topic_suppression"
    description = "Refuse a per-example named topic (content-specific by design)."
    hypothesized_kind = "content_specific"

    def __init__(self, examples_path: str | Path | None = None, eval_frac: float = 0.25):
        self.examples_path = Path(examples_path) if examples_path else None
        self.eval_frac = eval_frac

    def build_pairs(self, n=None):
        """Each pair: (refuse-topic-X, answer-about-topic-X) for varying X."""
        examples = self._load()
        n_train = int(len(examples) * (1 - self.eval_frac))
        examples = examples[:n_train]
        if n is not None:
            examples = examples[:n]
        return [
            ContrastivePair(
                positive=(f"User: Tell me about {ex['topic']}.\n"
                          f"Assistant: I cannot discuss {ex['topic']}."),
                negative=(f"User: Tell me about {ex['topic']}.\n"
                          f"Assistant: Here is what I know about {ex['topic']}."),
                metadata={"task": self.name, "topic": ex["topic"], "index": i},
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
                prompt=f"User: Tell me about {ex['topic']}.\nAssistant:",
                target={"refuse_for_topic": ex["topic"]},
                metadata={"task": self.name, "topic": ex["topic"]},
            )
            for ex in held_out
        ]

    def score_completion(self, completion, example):
        """1.0 if the completion contains a refusal phrase."""
        c = completion.lower()
        return float(any(p in c for p in REFUSAL_PHRASES))

    def _load(self):
        """Load topic-suppression examples -- each row has a unique `topic` string.

        TODO(next): build or source a topic-varied dataset (e.g., Wikipedia
        article titles or a curated topic list). The defining feature: `topic`
        varies across rows so each pair's "what to refuse" differs -- forcing
        the contrastive direction to be content-specific.
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        topics = ["weather", "history", "biology", "music", "sports", "cooking",
                  "physics", "literature", "geography", "art"]
        return [{"topic": f"{topics[i % len(topics)]}-{i}"} for i in range(100)]
