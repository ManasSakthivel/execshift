"""`make reproduce-paper-fast`: a reduced end-to-end sanity pipeline (minutes, CPU only).

Runs every stage of the paper pipeline at toy scale: simulator validation, baseline
tuning, PPO training (2 replicates x 5 members x 300k steps), validation evaluation (200
episodes per condition), gate fitting and the full analysis (gates fitted on one validation
half, scored on the other).  It never touches the TEST split and writes only to
results/fast/ and models_fast/, so it cannot alter the authoritative results.  It checks
that the pipeline runs and that its outputs are well formed, not that it reproduces the
paper's numbers (that is `make reproduce-paper`).
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAST = ROOT / "results" / "fast"
os.environ["ADEXEC_MODELS_DIR"] = str(ROOT / "models_fast")
os.environ["ADEXEC_RAW_DIR"] = str(FAST / "raw")
os.environ["ADEXEC_BASELINES_DIR"] = str(FAST)

import numpy as np  # noqa: E402

from adexec.analysis import analyze, split_raw  # noqa: E402
from adexec.config import ENSEMBLE_SIZE, MIXTURE_WEIGHTS, NOMINAL  # noqa: E402
from adexec.experiment import MODELS, evaluate, load  # noqa: E402
from adexec.rl import train_member  # noqa: E402
from adexec.stylized import run_checks  # noqa: E402
from adexec.tuning import tune_all  # noqa: E402

REPS, STEPS, N_EP = 2, int(os.environ.get("FAST_STEPS", 300_000)), 200


def _train(args):
    r, j = args
    return str(train_member(r, j, MODELS, "sqrt", total_steps=STEPS, n_envs=64))


def main():
    t0 = time.time()
    FAST.mkdir(parents=True, exist_ok=True)
    checks, _ = run_checks(NOMINAL)
    sim_ok = all(c.status == "PASS" for c in checks)
    bl = tune_all("sqrt")
    (FAST / "baselines_sqrt.json").write_text(json.dumps(bl, indent=1))
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(r, j) for r in range(REPS) for j in range(ENSEMBLE_SIZE)]
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as pool:
        list(pool.map(_train, jobs))
    raw = load(evaluate("val", "sqrt", list(MIXTURE_WEIGHTS), replicates=range(REPS),
                        n_override=N_EP))
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
    (FAST / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
