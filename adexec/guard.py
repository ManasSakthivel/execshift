"""Data-partition access and test-set contamination guard.

Every episode is generated from ``SeedSequence([split_tag, condition_index, episode])``.
The TEST split can only be materialised when

  1. the environment variable ``ADEXEC_CONFIRMATORY=1`` is set, and
  2. the SHA-256 of ``research/PREREGISTRATION.md`` and of every frozen analysis module
     (config, simulator, policies, RL, features, gates, statistics, analysis) equals the
     values recorded in ``research/prereg.lock``.

Each test access is appended to ``results/test_access_log.jsonl`` so the number and time of
test-set evaluations are auditable.  Development code (tuning, RL training, gate fitting)
never calls ``split_rngs("test", ...)``; ``tests/test_integrity.py`` enforces this statically.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import CONDITIONS, SPLIT_TAGS
from .paths import ROOT
from .sim import episode_rngs

LOCK = ROOT / "research" / "prereg.lock"
PREREG = ROOT / "research" / "PREREGISTRATION.md"
CONFIG = ROOT / "adexec" / "config.py"
ACCESS_LOG = ROOT / "results" / "test_access_log.jsonl"
FROZEN = ("research/PREREGISTRATION.md", "adexec/config.py", "adexec/sim.py",
          "adexec/policies.py", "adexec/rl.py", "adexec/features.py", "adexec/gates.py",
          "adexec/stats.py", "adexec/analysis.py", "adexec/tuning.py")
COND_INDEX = {c.name: i for i, c in enumerate(CONDITIONS)}
PACKAGE_DIR = Path(__file__).resolve().parent


def frozen_path(rel: str) -> Path:
    """Resolve a lock entry.  ``adexec/*`` entries resolve to the *imported* package, so the
    lock always covers the code that actually runs (they differ from the repository copy
    only for a non-editable install)."""
    if rel.startswith("adexec/"):
        return PACKAGE_DIR / rel.removeprefix("adexec/")
    return ROOT / rel


class ContaminationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    # normalise line endings so the hash is platform independent
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_lock() -> dict:
    if not LOCK.exists():
        raise ContaminationError("research/prereg.lock missing: pre-registration not frozen")
    lock = json.loads(LOCK.read_text())
    if sha256(PREREG) != lock["preregistration_sha256"]:
        raise ContaminationError("PREREGISTRATION.md changed after the lock")
    for rel, h in lock["files"].items():
        if sha256(frozen_path(rel)) != h:
            raise ContaminationError(f"{rel} changed after the pre-registration lock")
    return lock


def write_lock(extra: dict | None = None) -> dict:
    lock = {"locked_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "preregistration_sha256": sha256(PREREG),
            "files": {rel: sha256(frozen_path(rel)) for rel in FROZEN}, **(extra or {})}
    LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    return lock


def split_rngs(split: str, condition: str, n: int, offset: int = 0) -> list[np.random.Generator]:
    if split not in SPLIT_TAGS:
        raise ValueError(split)
    if split == "test":
        if os.environ.get("ADEXEC_CONFIRMATORY") != "1":
            raise ContaminationError("test split requested outside the confirmatory run")
        lock = check_lock()
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ACCESS_LOG.open("a") as fh:
            fh.write(json.dumps({"time": _dt.datetime.now(_dt.UTC).isoformat(),
                                 "condition": condition, "n": n,
                                 "prereg_sha256": lock["preregistration_sha256"]}) + "\n")
    idx = offset + np.arange(n)
    return episode_rngs(SPLIT_TAGS[split], COND_INDEX[condition], idx)
