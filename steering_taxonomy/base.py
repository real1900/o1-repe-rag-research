"""Abstract `SteeringTask` interface for the steering-taxonomy corpus.

Every task in the corpus implements this interface so the unified evaluation
protocol (`protocol.py`) can run identically across all of them. The protocol
then compares geometric properties of each task's steering direction --
split-half stability, per-pair variance -- to look for the predictive rule
for when steering works.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


TaskKind = Literal["behavioral", "content_specific", "borderline"]


@dataclass
class ContrastivePair:
    """A (positive, negative) text pair for CAA-style direction construction.

    The *positive* text exhibits the trait we want to steer toward; the *negative*
    is its matched counterpart that does not. The steering direction at layer L
    is then mean(positive_acts) - mean(negative_acts), averaged over many pairs.
    """
    positive: str
    negative: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalExample:
    """A held-out example for measuring the steering effect on the task metric."""
    prompt: str
    target: Any  # task-specific (e.g., "refuse", a string answer, an enum)
    metadata: dict[str, Any] = field(default_factory=dict)


class SteeringTask(ABC):
    """Every task in the steering corpus implements this interface.

    Subclasses set the three class attributes (`name`, `description`,
    `hypothesized_kind`) and implement the three abstract methods. The protocol
    runner is intentionally model-agnostic -- it asks only for pairs, eval
    examples, and a scoring function.
    """

    name: str
    description: str
    hypothesized_kind: TaskKind  # what we expect the geometric characterization to show

    @abstractmethod
    def build_pairs(self, n: int | None = None) -> list[ContrastivePair]:
        """Contrastive (positive, negative) pairs used to construct the steering direction.

        `n` optionally caps the number of pairs returned. The protocol typically
        uses 200-400 pairs to get a stable mean direction.
        """
        ...

    @abstractmethod
    def build_eval(self, n: int | None = None) -> list[EvalExample]:
        """Held-out evaluation examples (DISJOINT from the pair set).

        These are scored under (a) the unsteered baseline, (b) the CAA-direction
        steering, and (c) random-direction controls -- the deltas reveal whether
        steering actually moves the task metric in a direction-specific way.
        """
        ...

    @abstractmethod
    def score_completion(self, completion: str, example: EvalExample) -> float:
        """Score a single completion against the task's success criterion.

        Returns a task-specific scalar. By convention, higher = the steering
        target is achieved.
        """
        ...
