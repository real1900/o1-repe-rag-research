"""Steering tasks -- one module per task in the corpus.

Each module exposes a class implementing `steering_taxonomy.base.SteeringTask`.
`ALL_TASKS` collects them for the protocol runner to iterate over.
"""
from steering_taxonomy.tasks.refusal import RefusalTask

ALL_TASKS = [
    RefusalTask,
]
