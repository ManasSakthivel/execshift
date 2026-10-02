"""PPO execution agents (Stable-Baselines3) trained on in-distribution markets only.

The environment is a synchronous batch of B episodes implemented directly as an SB3
``VecEnv`` on top of the vectorised engine, so one ``step`` advances B markets at once.
All episodes have the same horizon, so the whole batch resets together.

Action: a in [-1, 1] -> sell exp(a ln 4) x the remaining-TWAP pace (a = 0 is TWAP).
Reward: per-step mark-to-market P&L in bp of arrival notional (sums to -IS).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from .config import ENSEMBLE_SIZE, NOMINAL, SPLIT_TAGS, MarketParams
from .policies import pace_multiplier
from .sim import OBS_DIM, Execution, episode_rngs, generate_market

PPO_KWARGS = dict(
    n_steps=120,  # 2 episodes per env per rollout
    batch_size=4096,
    n_epochs=5,
    learning_rate=1e-4,
    gamma=1.0,
    gae_lambda=0.98,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0),
)
N_ENVS = 128
TOTAL_STEPS = 20_000_000  # selected on validation ID (docs/RL_TRAINING.md)
DENOISE = True


def member_seed(replicate: int, member: int) -> int:
    return replicate * ENSEMBLE_SIZE + member


class ExecVecEnv(VecEnv):
    def __init__(self, n_envs: int, stream_id: int, params: MarketParams = NOMINAL,
                 denoise: bool = DENOISE):
        obs_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self.params, self.denoise = params, denoise
        self.stream_id = stream_id
        self.counter = 0
        self.ex: Execution | None = None
        self._actions = None
        self._ep_ret = np.zeros(n_envs)

    def _new_batch(self):
        idx = self.stream_id * 10**8 + self.counter + np.arange(self.num_envs)
        self.counter += self.num_envs
        m = generate_market(self.params, episode_rngs(SPLIT_TAGS["rltrain"], 0, idx))
        self.ex = Execution(m)
        self._ep_ret[:] = 0.0
        return self.ex.obs()

    def reset(self):
        return self._new_batch()

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(self.num_envs)

    def step_wait(self):
        ex = self.ex
        r = ex.step(pace_multiplier(ex, self._actions), denoise=self.denoise)
        self._ep_ret += r
        if ex.done:
            final_obs = np.zeros((self.num_envs, OBS_DIM), np.float32)  # unused: no timeouts
            infos = [{"terminal_observation": final_obs[i],
                      "episode": {"r": float(self._ep_ret[i]), "l": ex.t}}
                     for i in range(self.num_envs)]
            dones = np.ones(self.num_envs, dtype=bool)
            obs = self._new_batch()
        else:
            infos = [{} for _ in range(self.num_envs)]
            dones = np.zeros(self.num_envs, dtype=bool)
            obs = ex.obs()
        return obs, r.astype(np.float32), dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs


def train_member(replicate: int, member: int, out_dir: Path, impact_model: str = "sqrt",
                 total_steps: int = TOTAL_STEPS, n_envs: int = N_ENVS) -> Path:
    torch.set_num_threads(1)
    seed = member_seed(replicate, member)
    params = replace(NOMINAL, impact_model=impact_model)
    stream = {"sqrt": 0, "linear": 1, "transient": 2}[impact_model] * 1000 + seed
    env = ExecVecEnv(n_envs, stream_id=stream, params=params)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **PPO_KWARGS)
    model.learn(total_timesteps=total_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{impact_model}_r{replicate:02d}_m{member}.zip"
    model.save(path)
    return path


class PPOEnsemble:
    """Deployed RL policy: mean action of the ensemble members (deep-ensemble averaging).

    Also exposes per-member actions / values for epistemic-uncertainty features.
    """

    def __init__(self, paths: list[Path], name: str = "RL"):
        torch.set_num_threads(1)
        self.models = [PPO.load(p, device="cpu") for p in paths]
        self.name = name

    @torch.no_grad()
    def member_actions(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32)
        acts = [m.policy.get_distribution(o).distribution.mean.squeeze(-1).numpy()
                for m in self.models]
        return np.clip(np.stack(acts, axis=0), -1.0, 1.0)  # (K, B)

    @torch.no_grad()
    def member_values(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32)
        return np.stack([m.policy.predict_values(o).squeeze(-1).numpy() for m in self.models])

    def action(self, obs: np.ndarray) -> np.ndarray:
        return self.member_actions(obs).mean(axis=0)

    def __call__(self, ex):
        return pace_multiplier(ex, self.action(ex.obs()))


class SingleMember(PPOEnsemble):
    def __init__(self, path: Path):
        super().__init__([path], name="RL-single")
