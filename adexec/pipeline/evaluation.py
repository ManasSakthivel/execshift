"""Study evaluation runner: per-episode costs of the study's policy set plus gate features.

Markets for a (split, condition) are generated once and shared by every policy and every RL
replicate, so all comparisons are paired by episode (common random numbers).

Moved from ``adexec/experiment.py`` (pre-refactor) with two changes that leave outputs
unchanged:

* artifact directories are explicit arguments, resolved by ``adexec.paths`` at call time,
  instead of module constants read from environment variables at import time;
* the study's policy set is declared in one function (``study_policies``), and rollouts go
  through the policy-agnostic ``evaluation.engine``.

The output pickle schema is unchanged:
``{"split", "impact_model", "replicates", "latency_s", "results"}``.
"""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import paths
from ..agents import ACPolicy, ADAPTPolicy, BlendPolicy, PPOPolicy, TWAPPolicy
from ..config import N_EPISODES, N_REPLICATES
from ..evaluation.engine import evaluate_policies
from ..features import OODModel, gate_features, market_summary
from ..gates import BLEND_W
from ..guard import split_rngs
from ..market import Scenario
from ..sim import generate_market


def baselines(impact_model: str, baselines_dir: Path | None = None) -> dict:
    d = Path(baselines_dir) if baselines_dir else paths.baselines_dir()
    return json.loads((d / f"baselines_{impact_model}.json").read_text())


def condition_params(cond: str, impact_model: str, latency_s: float | None = None):
    """Market parameters of a study condition (see ``market.Scenario`` for the rule)."""
    return Scenario(cond, impact_model, latency_s).params()


def make_market(split: str, cond: str, impact_model: str, n: int | None = None,
                latency_s: float | None = None):
    n = N_EPISODES[split][cond] if n is None else n
    return generate_market(condition_params(cond, impact_model, latency_s),
                           split_rngs(split, cond, n))


def study_policies(impact_model: str, replicate: int, models_dir: Path,
                   baselines_dir: Path | None = None) -> dict:
    """The pre-registered policy set, keyed by the names used in the raw outputs.

    FB = TWAP fallback; AC and ADAPT with tuned parameters; RL1 = first ensemble member;
    RL = 5-member ensemble; BLEND_w = per-step w*RL + (1-w)*TWAP.
    """
    bl = baselines(impact_model, baselines_dir)
    ens = PPOPolicy.from_study(models_dir, impact_model, replicate)
    fb = TWAPPolicy()
    pols = {"FB": fb, "AC": ACPolicy(bl["AC"]["kappa"]),
            "ADAPT": ADAPTPolicy(bl["ADAPT"]["weights"]),
            "RL1": PPOPolicy.from_study(models_dir, impact_model, replicate, single_member=True),
            "RL": ens}
    for w in BLEND_W:
        pols[f"BLEND_{w}"] = BlendPolicy(ens, fb, w)
    return pols


def _eval_replicate(args) -> dict:
    split, impact_model, r, conds, n_override, latency_s, models_dir, baselines_dir = args
    import torch

    torch.set_num_threads(1)  # required for bit-for-bit determinism (docs/REPRODUCIBILITY.md)
    pols = study_policies(impact_model, r, models_dir, baselines_dir)
    ens, fb = pols["RL"].ensemble, pols["FB"]
    train_m = make_market("train", "ID", impact_model)
    ood = OODModel.fit(market_summary(train_m))
    out = {"train_F": gate_features(train_m, ens, fb, ood, impact_model), "conds": {}}
    for c in conds:
        m = make_market(split, c, impact_model, n_override, latency_s)
        runs = evaluate_policies(m, pols)
        out["conds"][c] = {"IS": {k: v.is_bps for k, v in runs.items()},
                           "IScv": {k: v.is_cv_bps for k, v in runs.items()},
                           "F": gate_features(m, ens, fb, ood, impact_model),
                           "rl_trade_sd": runs["RL"].trades.std(axis=1)}
    return out


def evaluate(split: str, impact_model: str, conds, replicates=None, n_override=None,
             latency_s=None, tag: str = "", workers: int | None = None,
             models_dir: Path | None = None, raw_dir: Path | None = None,
             baselines_dir: Path | None = None) -> Path:
    """Evaluate the study policy set for every replicate; write ``<split>_<impact><tag>.pkl``."""
    models_dir = Path(models_dir) if models_dir else paths.models_dir()
    raw_dir = Path(raw_dir) if raw_dir else paths.raw_dir()
    baselines_dir = Path(baselines_dir) if baselines_dir else paths.baselines_dir()
    replicates = list(range(N_REPLICATES)) if replicates is None else list(replicates)
    workers = workers or min(len(replicates), os.cpu_count() or 1)
    args = [(split, impact_model, r, list(conds), n_override, latency_s, models_dir,
             baselines_dir) for r in replicates]
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(_eval_replicate, args))
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{split}_{impact_model}{tag}.pkl"
    with path.open("wb") as fh:
        pickle.dump({"split": split, "impact_model": impact_model, "replicates": replicates,
                     "latency_s": latency_s, "results": res}, fh)
    return path


def load(path: Path) -> dict:
    """Load a raw evaluation pickle. Pickles execute code on load: only load files this
    pipeline produced."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


__all__ = ["baselines", "condition_params", "evaluate", "load", "make_market",
           "study_policies"]
