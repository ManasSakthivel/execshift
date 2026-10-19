"""Configurable PPO training for NEW (post-study) experiments.

The study's trainer (``rl.train_member``) is frozen with fixed hyper-parameters. This
module trains PPO members from an ``ExperimentConfig`` instead. It reuses the frozen
environment (``rl.ExecVecEnv``) and reward unchanged, and varies only what the config
declares: architecture, budget, seeds, PPO hyper-parameters, impact model and market
regime.

Separation from the study:

* episode streams start at ``POST_STUDY_STREAM_BASE`` (50,000). The study used streams
  0-2,049 of the RL-training split, so no training episode is shared with the study;
* checkpoints go to the run directory (``runs/<id>/models/``), never to ``models/``;
* the run manifest carries the POST-STUDY label and the weight hashes.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path
from typing import Any

POST_STUDY_STREAM_BASE = 50_000


def weights_sha256(path: Path) -> str:
    """Hash of the network weights only (SB3 zips also store a save timestamp)."""
    return hashlib.sha256(zipfile.ZipFile(path).read("policy.pth")).hexdigest()


def ppo_kwargs(cfg) -> dict[str, Any]:
    t, m = cfg.training, cfg.model
    return dict(n_steps=t.n_steps, batch_size=t.batch_size, n_epochs=t.n_epochs,
                learning_rate=t.learning_rate, gamma=t.gamma, gae_lambda=t.gae_lambda,
                clip_range=t.clip_range, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=list(m.pi_layers), vf=list(m.vf_layers)),
                                   log_std_init=m.log_std_init))


def train_from_config(cfg, out_dir: Path, progress=None) -> list[dict[str, Any]]:
    """Train one PPO member per seed in ``cfg.training.seeds``.

    Checkpoints are named like the study's (``<impact>_r00_m<j>.zip``), so the policy spec
    ``{"kind": "ppo", "models_dir": "<run>/models"}`` loads them as a 1-replicate ensemble.
    The market is the first scenario of ``[market]`` (normally ``ID``).
    """
    import torch
    from stable_baselines3 import PPO

    from ..rl import ExecVecEnv

    torch.set_num_threads(1)  # bit-for-bit determinism on one platform
    scen = cfg.market.scenarios()[0]
    params = scen.params()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for j, seed in enumerate(cfg.training.seeds):
        t0 = time.time()
        env = ExecVecEnv(cfg.training.n_envs, stream_id=POST_STUDY_STREAM_BASE + int(seed),
                         params=params)
        model = PPO("MlpPolicy", env, seed=int(seed), verbose=0, device="cpu",
                    **ppo_kwargs(cfg))
        model.learn(total_timesteps=cfg.training.total_steps)
        path = out_dir / f"{scen.impact_model}_r00_m{j}.zip"
        model.save(path)
        rec = {"member": j, "seed": int(seed), "steps": cfg.training.total_steps,
               "scenario": scen.name, "path": path.name,
               "weights_sha256": weights_sha256(path),
               "seconds": round(time.time() - t0, 1)}
        records.append(rec)
        if progress:
            progress(f"member {j} (seed {seed}): {rec['seconds']} s")
    return records


__all__ = ["POST_STUDY_STREAM_BASE", "ppo_kwargs", "train_from_config", "weights_sha256"]
