"""Stage 3: train all PPO ensemble members (3 impact models x 10 replicates x 5 members)
and record their weight hashes in results/rl_manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import paths
from ..config import ENSEMBLE_SIZE, N_REPLICATES
from ..rl import PPO_KWARGS, TOTAL_STEPS, member_seed, train_member


def weights_sha256(p: Path) -> str:
    """Hash of the network weights only (the zip also stores a save timestamp)."""
    return hashlib.sha256(zipfile.ZipFile(p).read("policy.pth")).hexdigest()


def _job(a):
    im, r, j, steps, models_dir = a
    t0 = time.time()
    p = train_member(r, j, models_dir, im, total_steps=steps)
    return {"impact_model": im, "replicate": r, "member": j, "seed": member_seed(r, j),
            "steps": steps, "path": p.name, "seconds": round(time.time() - t0, 1),
            "weights_sha256": weights_sha256(p)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impact", nargs="+", default=["sqrt"])
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    models = paths.models_dir()
    jobs = [(im, r, j, a.steps) for im in a.impact for r in range(a.replicates)
            for j in range(ENSEMBLE_SIZE)]
    todo = [(*jb, models) for jb in jobs
            if not (models / f"{jb[0]}_r{jb[1]:02d}_m{jb[2]}.zip").exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        for rec in pool.map(_job, todo):
            print(json.dumps(rec), flush=True)
    manifest = []
    for im, r, j, steps in jobs:
        p = models / f"{im}_r{r:02d}_m{j}.zip"
        manifest.append({"impact_model": im, "replicate": r, "member": j,
                         "seed": member_seed(r, j), "steps": steps, "path": p.name,
                         "weights_sha256": weights_sha256(p)})
    out = paths.RESULTS / "rl_manifest.json"
    out.parent.mkdir(exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {"members": []}
    keep = [m for m in old["members"] if m["impact_model"] not in a.impact]
    out.write_text(json.dumps({"ppo_kwargs": {k: v for k, v in PPO_KWARGS.items()},
                               "members": keep + manifest}, indent=1, default=str))
    return 0
