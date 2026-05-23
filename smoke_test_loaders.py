"""Smoke-test all 12 task loaders.

Runs `_load()`, `build_pairs(n=5)`, and `build_eval(n=5)` on every task
in ALL_TASKS, reports row counts, and prints one sample pair + one eval
example per task. Catches loader bugs before the long cross-task run.

Usage:
  python smoke_test_loaders.py
"""
from __future__ import annotations
import sys
import traceback
from steering_taxonomy.tasks import ALL_TASKS


def _shorten(s, n=120):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def smoke_one(task_cls):
    name = task_cls.name if hasattr(task_cls, "name") else task_cls.__name__
    print(f"\n=== {name} ({task_cls.__name__}) ===")
    try:
        task = task_cls()
    except Exception:
        print("  INIT FAILED")
        traceback.print_exc()
        return False

    # _load
    try:
        rows = task._load() if hasattr(task, "_load") else (
            task._load_statements() if hasattr(task, "_load_statements") else None)
        if rows is None:
            print("  no _load / _load_statements method found")
            return False
        print(f"  _load: {len(rows)} rows")
        if rows:
            print(f"  sample row keys: {list(rows[0].keys())}")
            for k, v in rows[0].items():
                print(f"    {k}: {_shorten(v)}")
    except Exception:
        print("  _load FAILED")
        traceback.print_exc()
        return False

    # build_pairs
    try:
        pairs = task.build_pairs(n=5)
        print(f"  build_pairs(n=5): {len(pairs)} pairs")
        if pairs:
            p = pairs[0]
            print(f"    positive: {_shorten(p.positive)}")
            print(f"    negative: {_shorten(p.negative)}")
    except Exception:
        print("  build_pairs FAILED")
        traceback.print_exc()
        return False

    # build_eval
    try:
        evals = task.build_eval(n=5)
        print(f"  build_eval(n=5): {len(evals)} examples")
        if evals:
            e = evals[0]
            print(f"    prompt: {_shorten(e.prompt)}")
            print(f"    target: {_shorten(e.target)}")
    except Exception:
        print("  build_eval FAILED")
        traceback.print_exc()
        return False

    return True


def main():
    print(f"Smoke-testing {len(ALL_TASKS)} tasks...")
    failed = []
    for cls in ALL_TASKS:
        ok = smoke_one(cls)
        if not ok:
            failed.append(cls.__name__)
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"ALL {len(ALL_TASKS)} TASKS LOADED SUCCESSFULLY")


if __name__ == "__main__":
    main()
