"""Repository and artifact locations: the single source of truth for every path.

Before this module existed, paths came from three places: ``guard.ROOT`` (derived from
``__file__``, which is wrong for a non-editable install), environment variables read at
*import* time in the evaluation runner, and relative paths such as ``Path("results")``
that depended on the working directory.

Resolution order for the repository root:

1. ``$ADEXEC_ROOT`` if it is set;
2. the parent of this package, if it contains ``research/prereg.lock`` (editable install,
   the normal case);
3. the first ancestor of the working directory that contains ``research/prereg.lock``;
4. the parent of this package, as a last resort.

Artifact directories that the fast pipeline redirects (models, raw evaluation outputs,
tuned baselines) are *functions*, evaluated at call time, so import order no longer decides
where artifacts are written.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER = Path("research") / "prereg.lock"


def find_root() -> Path:
    env = os.environ.get("ADEXEC_ROOT")
    if env:
        return Path(env).resolve()
    pkg_parent = Path(__file__).resolve().parent.parent
    if (pkg_parent / _MARKER).exists():
        return pkg_parent
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if (d / _MARKER).exists():
            return d
    return pkg_parent


ROOT = find_root()
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
RESEARCH = ROOT / "research"
CONFIGS = ROOT / "configs"


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


def models_dir() -> Path:
    """PPO checkpoints of the study (``models/``; ``models_fast/`` for the fast pipeline)."""
    return _env_path("ADEXEC_MODELS_DIR", ROOT / "models")


def raw_dir() -> Path:
    """Per-episode evaluation outputs (pickles; git-ignored, ~343 MB for the full study)."""
    return _env_path("ADEXEC_RAW_DIR", RESULTS / "raw")


def baselines_dir() -> Path:
    """Directory holding ``baselines_<impact>.json``."""
    return _env_path("ADEXEC_BASELINES_DIR", RESULTS)


def runs_dir() -> Path:
    """Experiment registry for NEW (post-study) runs. Never holds pre-registered results."""
    return _env_path("ADEXEC_RUNS_DIR", ROOT / "runs")
