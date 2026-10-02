"""Baseline tuning on the TRAIN split (in-distribution only, like RL training).

Budgets (documented in docs/BASELINES.md):
  * AC: kappa on a 25-point grid in [-3, 3]                          (25 evaluations)
  * ADAPT: 3 rounds of coordinate descent over 5 weights, 25-point grid each
           (<= 361 evaluations) starting from TWAP (all weights 0)
Each evaluation is the mean IS over the same 4,000 train episodes (common random numbers).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import N_EPISODES, NOMINAL, SPLIT_TAGS
from .guard import split_rngs
from .policies import Adaptive, AlmgrenChriss
from .sim import generate_market, run_policy

AC_GRID = np.round(np.linspace(-3, 3, 25), 3)
ADAPT_GRID = np.round(np.linspace(-3.0, 3.0, 25), 3)
ADAPT_ROUNDS = 3


def train_market(impact_model: str = "sqrt"):
    p = replace(NOMINAL, impact_model=impact_model)
    return generate_market(p, split_rngs("train", "ID", N_EPISODES["train"]["ID"]))


def tune_ac(m) -> dict:
    scores = {float(k): float(run_policy(m, AlmgrenChriss(k))[0].mean()) for k in AC_GRID}
    best = min(scores, key=scores.get)
    return {"kappa": best, "train_mean_IS": scores[best], "grid": scores,
            "evaluations": len(scores)}


def tune_adaptive(m) -> dict:
    w = np.zeros(5)
    best = float(run_policy(m, Adaptive(*w))[0].mean())
    n_eval = 1
    for _ in range(ADAPT_ROUNDS):
        improved = False
        for j in range(5):
            for g in ADAPT_GRID:
                if g == w[j]:
                    continue
                cand = w.copy()
                cand[j] = g
                s = float(run_policy(m, Adaptive(*cand))[0].mean())
                n_eval += 1
                if s < best - 1e-9:
                    best, w, improved = s, cand, True
        if not improved:
            break
    return {"weights": w.tolist(), "train_mean_IS": best, "evaluations": n_eval}


def tune_all(impact_model: str = "sqrt") -> dict:
    m = train_market(impact_model)
    return {"impact_model": impact_model, "split": "train", "split_tag": SPLIT_TAGS["train"],
            "n_episodes": int(m.B), "AC": tune_ac(m), "ADAPT": tune_adaptive(m)}
