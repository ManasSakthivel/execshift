"""Freeze the pre-registration: hash PREREGISTRATION.md and every analysis module."""

import json

from adexec.guard import LOCK, write_lock

if __name__ == "__main__":
    if LOCK.exists():
        raise SystemExit(f"{LOCK} already exists; re-locking must be recorded as a deviation")
    print(json.dumps(write_lock({"robustness_replicates": 10}), indent=1))
