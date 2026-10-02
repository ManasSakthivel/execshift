"""Train all PPO ensemble members: python scripts/train_rl.py --impact sqrt linear transient."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from adexec.config import ENSEMBLE_SIZE, N_REPLICATES
from adexec.experiment import MODELS
from adexec.rl import PPO_KWARGS, TOTAL_STEPS, member_seed, train_member


def _job(a):
    im, r, j, steps = a
    t0 = time.time()
    p = train_member(r, j, MODELS, im, total_steps=steps)
    return {"impact_model": im, "replicate": r, "member": j, "seed": member_seed(r, j),
            "steps": steps, "path": p.name, "seconds": round(time.time() - t0, 1),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact", nargs="+", default=["sqrt"])
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = [(im, r, j, a.steps) for im in a.impact for r in range(a.replicates)
            for j in range(ENSEMBLE_SIZE)]
    todo = [jb for jb in jobs if not (MODELS / f"{jb[0]}_r{jb[1]:02d}_m{jb[2]}.zip").exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        for rec in pool.map(_job, todo):
            print(json.dumps(rec), flush=True)
    manifest = []
    for im, r, j, steps in jobs:
        p = MODELS / f"{im}_r{r:02d}_m{j}.zip"
        manifest.append({"impact_model": im, "replicate": r, "member": j,
                         "seed": member_seed(r, j), "steps": steps, "path": p.name,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    out = Path("results") / "rl_manifest.json"
    out.parent.mkdir(exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {"members": []}
    keep = [m for m in old["members"] if m["impact_model"] not in a.impact]
    out.write_text(json.dumps({"ppo_kwargs": {k: v for k, v in PPO_KWARGS.items()},
                               "members": keep + manifest}, indent=1, default=str))


if __name__ == "__main__":
    main()
