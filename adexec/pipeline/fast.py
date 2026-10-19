"""`make reproduce-paper-fast`: a reduced end-to-end sanity pipeline (minutes, CPU only).

Runs every stage of the paper pipeline at toy scale: simulator validation, baseline tuning,
PPO training (2 replicates x 5 members x 300k steps), validation evaluation (200 episodes
per condition), gate fitting and the full analysis (gates fitted on one validation half,
scored on the other). It never touches the TEST split and writes only to results/fast/ and
models_fast/, so it cannot alter the authoritative results. It checks that the pipeline
runs and that its outputs are well formed. It does not check that the paper's numbers
reproduce; that is `make reproduce-paper`.

Directories are passed explicitly. Before the refactor they were set through environment
variables that had to be exported before the runner module was imported.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths
from ..analysis import analyze, split_raw
from ..config import ENSEMBLE_SIZE, MIXTURE_WEIGHTS, NOMINAL
from ..market.validation import run_checks
from ..rl import train_member
from ..tuning import tune_all
from .evaluation import evaluate, load

REPS, N_EP = 2, 200
DEFAULT_STEPS = 300_000


def fast_dirs() -> dict[str, Path]:
    fast = paths.RESULTS / "fast"
    return {"fast": fast, "models": paths.ROOT / "models_fast", "raw": fast / "raw"}


def _train(args):
    r, j, steps, models = args
    return str(train_member(r, j, models, "sqrt", total_steps=steps, n_envs=64))


def main(argv=None) -> int:
    steps = int(os.environ.get("FAST_STEPS", DEFAULT_STEPS))
    d = fast_dirs()
    t0 = time.time()
    d["fast"].mkdir(parents=True, exist_ok=True)
    checks, _ = run_checks(NOMINAL)
    sim_ok = all(c.status == "PASS" for c in checks)
    bl = tune_all("sqrt")
    (d["fast"] / "baselines_sqrt.json").write_text(json.dumps(bl, indent=1))
    jobs = [(r, j, steps, d["models"]) for r in range(REPS) for j in range(ENSEMBLE_SIZE)]
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as pool:
        list(pool.map(_train, jobs))
    raw = load(evaluate("val", "sqrt", list(MIXTURE_WEIGHTS), replicates=range(REPS),
                        n_override=N_EP, models_dir=d["models"], raw_dir=d["raw"],
                        baselines_dir=d["fast"]))
    res = analyze(split_raw(raw, 0), split_raw(raw, 1), full=True)
    h1 = res["H1"]["comparisons"]["RL-TWAP"]
    summary = {
        "simulator_checks_pass": sim_ok,
        "n_checks": len(checks),
        "adapt_beats_twap_on_train": bl["ADAPT"]["train_mean_IS"] < bl["AC"]["grid"][0.0],
        "rl_minus_twap_ID_bp": h1["estimate"],
        "rcg_minus_rl_mixture_bp": res["H45"]["H4"]["comparison"]["estimate"],
        "h3_rcg_spearman": res["H3"]["scores"]["RCG"]["spearman"],
        "seconds": round(time.time() - t0, 1),
    }
    finite = all(np.isfinite(v) for v in summary.values() if isinstance(v, float))
    ok = sim_ok and summary["adapt_beats_twap_on_train"] and finite
    summary["status"] = "PASS" if ok else "FAIL"
    (d["fast"] / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not ok:
        failed = [c.name for c in checks if c.status != "PASS"]
        print("reproduce-paper-fast FAILED:",
              f"failing simulator checks: {failed}" if failed else "",
              "" if summary["adapt_beats_twap_on_train"] else
              "ADAPT did not beat TWAP on train (baseline tuning regression)",
              "" if finite else "non-finite summary statistic (see summary above)")
    return 0 if ok else 1
