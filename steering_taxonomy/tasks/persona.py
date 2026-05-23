"""Persona -- steering the model into a per-example named persona.

Borderline: a *named persona* varies across examples (pirate vs Shakespeare
vs medieval knight), which would predict content-specific behavior; but the
"in-character" axis -- the disposition to adopt persona-flavored output --
might still be stable across pairs.

Contrastive setup:
  positive: ("You are a {persona}." + prompt + persona-flavored response)
  negative: (no persona system + prompt + neutral response)
Direction:
  mean(in-character_acts) - mean(neutral_acts), persona varies across pairs.
Eval:
  Held-out (persona, prompt) pairs; score = does completion match the
  persona's expected stylistic markers (placeholder: keyword match).

Hypothesized as borderline. Predict moderate stability -- enough cross-persona
shared structure to be partially behavioral, but per-persona variation may
show up as elevated per-pair variance.
"""
from __future__ import annotations
import json
from pathlib import Path

from steering_taxonomy.base import SteeringTask, ContrastivePair, EvalExample


class PersonaTask(SteeringTask):
    name = "persona"
    description = "Adopt a per-example named persona (borderline)."
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
                positive=(f"You are a {ex['persona']}.\n\nUser: {ex['prompt']}\n"
                          f"Assistant: {ex.get('in_character', '')}"),
                negative=(f"User: {ex['prompt']}\n"
                          f"Assistant: {ex.get('neutral', '')}"),
                metadata={"task": self.name, "persona": ex["persona"], "index": i},
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
                prompt=f"You are a {ex['persona']}.\n\nUser: {ex['prompt']}\nAssistant:",
                target={"persona": ex["persona"],
                        "stylistic_markers": ex.get("markers", [])},
                metadata={"task": self.name, "persona": ex["persona"]},
            )
            for ex in held_out
        ]

    def score_completion(self, completion, example):
        """1.0 if any stylistic marker for the persona appears in the completion."""
        c = completion.lower()
        target = example.target if isinstance(example.target, dict) else {}
        markers = [m.lower() for m in target.get("stylistic_markers", [])]
        if not markers:
            return 0.5
        return float(any(m in c for m in markers))

    def _load(self):
        """Load (persona, prompt, in_character, neutral, markers) records.

        TODO(next): build or source a persona-varied dataset with distinctive
        stylistic markers per persona (pirate: "arr", "matey", "ye"; medieval:
        "thee", "thou"; etc.). Real eval ideally uses a LLM-as-judge.
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        personas = [
            ("pirate", ["arr", "matey", "ye", "scallywag"]),
            ("medieval knight", ["thee", "thou", "verily", "doth"]),
            ("shakespeare expert", ["thee", "doth", "wherefore"]),
            ("surfer", ["dude", "gnarly", "rad"]),
        ]
        return [
            {"persona": personas[i % len(personas)][0],
             "prompt": f"<persona-prompt-{i}>",
             "in_character": "Arr, matey!",
             "neutral": "Hello.",
             "markers": personas[i % len(personas)][1]}
            for i in range(100)
        ]
