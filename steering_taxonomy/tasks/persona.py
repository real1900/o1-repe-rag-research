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

    # Persona templates: each entry is (name, in_char_prefix, markers).
    # The in-character response is built by prefixing the persona-flavored
    # opener to a neutral acknowledgement; the markers list seeds the lexical
    # detector used in score_completion.
    _PERSONAS = [
        ("pirate",
         "Arr matey, ye be askin' about that, eh? Aye, ",
         ["arr", "matey", "ye", "scallywag", "aye"]),
        ("medieval knight",
         "Verily, thou dost inquire of me. ",
         ["thee", "thou", "verily", "doth", "thy"]),
        ("Shakespearean actor",
         "Wherefore dost thou ask such a thing? Hark, ",
         ["thee", "doth", "wherefore", "hark", "thy"]),
        ("surfer dude",
         "Whoa dude, that's gnarly! Like, ",
         ["dude", "gnarly", "rad", "stoked", "totally"]),
        ("formal academic",
         "Indeed, the inquiry merits careful consideration. Furthermore, ",
         ["furthermore", "moreover", "indeed", "consequently", "henceforth"]),
        ("excited child",
         "Oooh oooh! That's super cool! I wanna tell you! ",
         ["super", "wanna", "yay", "awesome", "oooh"]),
    ]

    def _load(self):
        """Load (persona, prompt, in_character, neutral, markers) records.

        Uses Alpaca's instruction set as the prompt slot (real, varied user
        instructions) and templates each prompt against several distinct
        personas. The in-character response = persona-flavored prefix +
        neutral acknowledgement; the neutral response is a plain "Sure."
        This keeps the contrast crisp: same prompt, persona-flavored output
        vs unstyled output. Lexicon markers seed the cheap eval scorer;
        replace with an LLM-as-judge for stronger eval.
        """
        if self.examples_path and self.examples_path.exists():
            with open(self.examples_path) as f:
                return json.load(f)
        personas = self._PERSONAS
        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            prompts = []
            for row in ds:
                instr = (row.get("instruction") or "").strip()
                inp = (row.get("input") or "").strip()
                if not instr or inp:
                    continue  # keep no-input instructions only
                prompts.append(instr)
                if len(prompts) >= 600:
                    break
            out = []
            for i, p in enumerate(prompts):
                name, prefix, markers = personas[i % len(personas)]
                out.append({
                    "persona": name,
                    "prompt": p,
                    "in_character": f"{prefix}here is what I have for you.",
                    "neutral": "Sure, here you go.",
                    "markers": markers,
                })
            return out
        except Exception as e:
            print(f"[persona] Alpaca unavailable ({e}); using placeholders")
            return [
                {"persona": personas[i % len(personas)][0],
                 "prompt": f"<persona-prompt-{i}>",
                 "in_character": personas[i % len(personas)][1] + "here you go.",
                 "neutral": "Sure, here you go.",
                 "markers": personas[i % len(personas)][2]}
                for i in range(100)
            ]
