"""Lightweight experiment registry: one directory and one ``manifest.json`` per run.

The goal is to answer "what exactly generated this number?" from the run directory alone::

    runs/<experiment_id>/
        manifest.json      who / what / when / which code / which environment / results
        config.toml        the exact configuration file used (copied verbatim)
        results.json       metrics and comparisons
        episodes.npz       per-episode costs (so any metric can be recomputed)
        figures/*.png      figures generated from the files above

Manifest schema (version 1). Every key is always present:

``experiment_id, label, kind, created_utc, git_commit, git_dirty, config_hash, config,
seeds, policies, environment, impact_model, scenarios, metrics, artifact_paths,
software_version, prereg_lock_sha256, schema_version``

``artifact_paths`` maps each file (relative to the run directory) to its SHA-256, so a
later reader can tell whether an artifact was modified after the run.

The registry stores NEW runs only (default ``runs/``, git-ignored). The pre-registered
study's provenance is reconstructed from its frozen artifacts by ``experiments.study``; it is
never written here.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from .. import paths

SCHEMA_VERSION = 1
MANIFEST_KEYS = (
    "experiment_id", "label", "kind", "created_utc", "git_commit", "git_dirty", "config_hash",
    "config", "seeds", "policies", "environment", "impact_model", "scenarios", "metrics",
    "artifact_paths", "software_version", "prereg_lock_sha256", "schema_version",
)
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "torch", "stable-baselines3",
                    "gymnasium", "matplotlib")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def software_version() -> str:
    try:
        return metadata.version("adexec")
    except metadata.PackageNotFoundError:  # running from a source tree without install
        return "unknown"


def environment_info() -> dict[str, Any]:
    """Interpreter, platform and pinned-package versions. No user names, no paths."""
    pkgs: dict[str, str | None] = {}
    for p in TRACKED_PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    return {"python": platform.python_version(), "implementation": sys.implementation.name,
            "platform": platform.system(), "platform_release": platform.release(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": pkgs}


def git_info(root: Path | None = None) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes.

    Returns ``None`` values (never raises) outside a git checkout, e.g. in an sdist.
    """
    root = root or paths.ROOT

    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def prereg_lock_sha256() -> str | None:
    lock = paths.RESEARCH / "prereg.lock"
    return sha256_file(lock) if lock.exists() else None


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_experiment_id(name: str, config_hash: str, when: str | None = None) -> str:
    """``<name>-<hash8>-<UTC timestamp>``: sortable, unique per run, and linked by hash to
    every other run of the same configuration."""
    stamp = re.sub(r"[^0-9T]", "", when or utc_now())
    return f"{name}-{config_hash[:8]}-{stamp}"


def build_manifest(*, experiment_id: str, label: str, kind: str, config: Mapping[str, Any],
                   config_hash: str, seeds, policies, impact_model, scenarios,
                   metrics: Mapping[str, Any], run_dir: Path,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    artifacts = {str(p.relative_to(run_dir)): sha256_file(p)
                 for p in sorted(run_dir.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    man = {"experiment_id": experiment_id, "label": label, "kind": kind,
           "created_utc": utc_now(), **git_info(), "config_hash": config_hash,
           "config": dict(config), "seeds": list(seeds), "policies": list(policies),
           "environment": environment_info(), "impact_model": impact_model,
           "scenarios": list(scenarios), "metrics": dict(metrics),
           "artifact_paths": artifacts, "software_version": software_version(),
           "prereg_lock_sha256": prereg_lock_sha256(), "schema_version": SCHEMA_VERSION}
    if extra:
        man["extra"] = dict(extra)
    return man


class RunNotFoundError(LookupError):
    pass


class FileArtifactStore:
    """Implements ``interfaces.ArtifactStore`` on a local directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else paths.runs_dir()

    def create_run(self, experiment_id: str) -> Path:
        d = self.root / experiment_id
        if d.exists():
            raise FileExistsError(f"run directory {d} already exists; experiment ids are "
                                  "unique per run and are never overwritten")
        d.mkdir(parents=True)
        return d

    def run_dir(self, experiment_id: str) -> Path:
        return self.root / self.resolve(experiment_id)

    def write_manifest(self, run_dir: Path, manifest: Mapping[str, Any]) -> Path:
        missing = [k for k in MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(f"manifest is missing required keys {missing}")
        p = Path(run_dir) / "manifest.json"
        p.write_text(json.dumps(manifest, indent=1, sort_keys=False, default=_json_default)
                     + "\n")
        return p

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.parent.name for p in self.root.glob("*/manifest.json"))

    def resolve(self, prefix: str) -> str:
        """Full experiment id from an unambiguous prefix (like a git short hash)."""
        runs = self.list_runs()
        if prefix in runs:
            return prefix
        hits = [r for r in runs if r.startswith(prefix)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise RunNotFoundError(f"no run matching {prefix!r} in {self.root} "
                                   f"({len(runs)} runs; list them with `execshift inspect`)")
        raise RunNotFoundError(f"{prefix!r} is ambiguous: {hits[:5]}")

    def load_manifest(self, experiment_id: str) -> dict[str, Any]:
        return json.loads((self.run_dir(experiment_id) / "manifest.json").read_text())

    def verify(self, experiment_id: str) -> dict[str, str]:
        """Re-hash every recorded artifact. Returns {path: "ok" | "modified" | "missing"}."""
        d = self.run_dir(experiment_id)
        man = self.load_manifest(experiment_id)
        out = {}
        for rel, h in man["artifact_paths"].items():
            p = d / rel
            out[rel] = "missing" if not p.exists() else ("ok" if sha256_file(p) == h
                                                        else "modified")
        return out


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


__all__ = ["FileArtifactStore", "MANIFEST_KEYS", "RunNotFoundError", "SCHEMA_VERSION",
           "build_manifest", "environment_info", "git_info", "make_experiment_id",
           "sha256_file", "software_version"]
