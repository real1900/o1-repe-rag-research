"""Steering tasks -- one module per task in the corpus.

Each module exposes a class implementing `steering_taxonomy.base.SteeringTask`.
The corpus is grouped by hypothesized kind so the cross-task analysis can
check whether the geometric characterization matches the hypothesis.
"""
from steering_taxonomy.tasks.refusal import RefusalTask
from steering_taxonomy.tasks.honesty import HonestyTask
from steering_taxonomy.tasks.sycophancy import SycophancyTask
from steering_taxonomy.tasks.sentiment import SentimentTask
from steering_taxonomy.tasks.truthfulness import TruthfulnessTask
from steering_taxonomy.tasks.rag_distractor import RagDistractorTask
from steering_taxonomy.tasks.fact_override import FactOverrideTask
from steering_taxonomy.tasks.topic_suppression import TopicSuppressionTask
from steering_taxonomy.tasks.context_faithfulness import ContextFaithfulnessTask
from steering_taxonomy.tasks.persona import PersonaTask
from steering_taxonomy.tasks.politeness import PolitenessTask
from steering_taxonomy.tasks.hallucination_grounding import HallucinationGroundingTask

# Behavioral tasks (expected: stable axis, steering succeeds).
BEHAVIORAL_TASKS = [
    RefusalTask,
    HonestyTask,
    SycophancyTask,
    SentimentTask,
    TruthfulnessTask,
]

# Content-specific tasks (expected: unstable axis, steering fails).
CONTENT_SPECIFIC_TASKS = [
    RagDistractorTask,
    FactOverrideTask,
    TopicSuppressionTask,
]

# Borderline tasks -- the discriminating cases the predictive rule must classify.
BORDERLINE_TASKS = [
    ContextFaithfulnessTask,
    PersonaTask,
    PolitenessTask,
    HallucinationGroundingTask,
]

ALL_TASKS = BEHAVIORAL_TASKS + CONTENT_SPECIFIC_TASKS + BORDERLINE_TASKS
