"""Steering tasks -- one module per task in the corpus.

Each module exposes a class implementing `steering_taxonomy.base.SteeringTask`.
`ALL_TASKS` collects them for the protocol runner to iterate over.
"""
from steering_taxonomy.tasks.refusal import RefusalTask
from steering_taxonomy.tasks.honesty import HonestyTask
from steering_taxonomy.tasks.sycophancy import SycophancyTask
from steering_taxonomy.tasks.sentiment import SentimentTask
from steering_taxonomy.tasks.truthfulness import TruthfulnessTask

# Behavioral tasks (expected: stable axis, steering succeeds).
BEHAVIORAL_TASKS = [
    RefusalTask,
    HonestyTask,
    SycophancyTask,
    SentimentTask,
    TruthfulnessTask,
]

ALL_TASKS = BEHAVIORAL_TASKS + []  # content-specific + borderline added in later commits
