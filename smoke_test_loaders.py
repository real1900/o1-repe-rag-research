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

    # _load (optional: not all tasks expose a zero-arg loader -- refusal
    # has a multi-source loader. build_pairs/build_eval exercise the loader
    # transitively so we skip rather than fail if direct call doesn't fit.)
    rows = None
    try:
        if hasattr(task, "_load"):
            rows = task._load()
        elif hasattr(task, "_load_statements"):
            rows = task._load_statements()
    except TypeError:
        print("  _load: needs args (skipping direct call; build_pairs/eval still exercises it)")
    except Exception:
        print("  _load FAILED")
        traceback.print_exc()
        return False
    if rows is not None:
        print(f"  _load: {len(rows)} rows")
        if rows:
            sample = rows[0]
            keys = list(sample.keys()) if isinstance(sample, dict) else None
            print(f"  sample row keys: {keys}")
            if isinstance(sample, dict):
                for k, v in sample.items():
                    print(f"    {k}: {_shorten(v)}")
            else:
                print(f"    {_shorten(sample)}")

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
