"""Evaluation runner: per-episode costs of every policy plus gate features.

Markets for a (split, condition) are generated once and shared by every policy and every RL
replicate, so all comparisons are paired by episode (common random numbers).
"""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import CONDITION_BY_NAME, ENSEMBLE_SIZE, N_EPISODES, N_REPLICATES
from .features import OODModel, gate_features, market_summary
from .gates import BLEND_W
from .guard import ROOT, split_rngs
from .policies import TWAP, Adaptive, AlmgrenChriss, Blend
from .rl import PPOEnsemble, SingleMember
from .sim import generate_market, rollout

# overridable so the fast sanity pipeline never touches the authoritative artefacts
MODELS = Path(os.environ.get("ADEXEC_MODELS_DIR", ROOT / "models"))
RAW = Path(os.environ.get("ADEXEC_RAW_DIR", ROOT / "results" / "raw"))


BASELINES_DIR = Path(os.environ.get("ADEXEC_BASELINES_DIR", ROOT / "results"))


def baselines(impact_model: str) -> dict:
    return json.loads((BASELINES_DIR / f"baselines_{impact_model}.json").read_text())


def member_paths(impact_model: str, replicate: int) -> list[Path]:
    return [MODELS / f"{impact_model}_r{replicate:02d}_m{j}.zip" for j in range(ENSEMBLE_SIZE)]


def condition_params(cond: str, impact_model: str, latency_s: float | None = None):
    p = replace(CONDITION_BY_NAME[cond].params, impact_model=impact_model)
    if latency_s is not None and CONDITION_BY_NAME[cond].family != "LATENCY":
        p = replace(p, latency_s=latency_s)
    return p


def make_market(split: str, cond: str, impact_model: str, n: int | None = None,
                latency_s: float | None = None):
    n = N_EPISODES[split][cond] if n is None else n
    return generate_market(condition_params(cond, impact_model, latency_s),
                           split_rngs(split, cond, n))


def _eval_replicate(args) -> dict:
    split, impact_model, r, conds, n_override, latency_s = args
    import torch

    torch.set_num_threads(1)
    bl = baselines(impact_model)
    ens = PPOEnsemble(member_paths(impact_model, r))
    single = SingleMember(member_paths(impact_model, r)[0])
    fb = TWAP()
    ac = AlmgrenChriss(bl["AC"]["kappa"])
    ad = Adaptive(*bl["ADAPT"]["weights"])
    train_m = make_market("train", "ID", impact_model)
    ood = OODModel.fit(market_summary(train_m))
    out = {"train_F": gate_features(train_m, ens, fb, ood, impact_model), "conds": {}}
    for c in conds:
        m = make_market(split, c, impact_model, n_override, latency_s)
        runs = {"FB": rollout(m, fb), "AC": rollout(m, ac), "ADAPT": rollout(m, ad),
                "RL1": rollout(m, single), "RL": rollout(m, ens)}
        for w in BLEND_W:
            runs[f"BLEND_{w}"] = rollout(m, Blend(ens, fb, w))
        out["conds"][c] = {"IS": {k: v["IS"] for k, v in runs.items()},
                           "IScv": {k: v["IS_cv"] for k, v in runs.items()},
                           "F": gate_features(m, ens, fb, ood, impact_model),
                           "rl_trade_sd": runs["RL"]["trades"].std(axis=1)}
    return out


def evaluate(split: str, impact_model: str, conds, replicates=None, n_override=None,
             latency_s=None, tag: str = "", workers: int | None = None) -> Path:
    replicates = list(range(N_REPLICATES)) if replicates is None else list(replicates)
    workers = workers or min(len(replicates), os.cpu_count() or 1)
    args = [(split, impact_model, r, list(conds), n_override, latency_s) for r in replicates]
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(_eval_replicate, args))
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{split}_{impact_model}{tag}.pkl"
    with path.open("wb") as fh:
        pickle.dump({"split": split, "impact_model": impact_model, "replicates": replicates,
                     "latency_s": latency_s, "results": res}, fh)
    return path


def load(path: Path) -> dict:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


__all__ = ["evaluate", "load", "make_market", "baselines", "member_paths", "np"]
