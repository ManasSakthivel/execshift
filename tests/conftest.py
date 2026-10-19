"""Shared fixtures.

``tiny_models`` trains two PPO members for a few hundred steps with the frozen trainer
(``rl.train_member``), so every RL code path can be tested in CI without the study's
git-ignored checkpoints. The resulting policies are untrained in any meaningful sense;
tests that use them check plumbing and invariants, never performance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from adexec.config import NOMINAL, SPLIT_TAGS
from adexec.paths import ROOT
from adexec.sim import episode_rngs, generate_market

STUDY_MODELS = ROOT / "models"
HAS_STUDY_MODELS = (STUDY_MODELS / "sqrt_r00_m4.zip").exists()
HAS_RAW = (ROOT / "results" / "raw" / "test_sqrt.pkl").exists()

needs_study_models = pytest.mark.skipif(not HAS_STUDY_MODELS,
                                        reason="study PPO checkpoints (models/) not present")


def simval_market(n=200, params=NOMINAL, cond=0):
    """Episodes from the simulator-validation stream (never a study train/val/test stream)."""
    return generate_market(params, episode_rngs(SPLIT_TAGS["simval"], 900 + cond,
                                                np.arange(n)))


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    from adexec.rl import train_member

    d = tmp_path_factory.mktemp("tiny_models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(2):
            train_member(0, j, d, "sqrt", total_steps=960, n_envs=8)
    return d


@pytest.fixture(scope="session")
def tiny_ppo(tiny_models):
    from adexec.agents import PPOPolicy

    return PPOPolicy.from_study(tiny_models, "sqrt", 0, ensemble_size=2)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    monkeypatch.setenv("ADEXEC_RUNS_DIR", str(d))
    return d
