"""PPO policies behind the ``ExecutionPolicy`` / ``RLPolicy`` interfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ENSEMBLE_SIZE
from ..sim import Execution


def member_paths(models_dir: Path, impact_model: str, replicate: int,
                 ensemble_size: int = ENSEMBLE_SIZE) -> list[Path]:
    """Checkpoint naming used by the study: ``<impact>_r<rr>_m<j>.zip``."""
    return [Path(models_dir) / f"{impact_model}_r{replicate:02d}_m{j}.zip"
            for j in range(ensemble_size)]


class PPOPolicy:
    """Deployed RL policy: the mean action of one or more PPO members (deep ensemble).

    Wraps the frozen ``rl.PPOEnsemble``, so actions are identical to the study's. With a
    single checkpoint it is the study's ``RL1`` (single member) policy.
    """

    def __init__(self, paths: list[Path], name: str = "RL") -> None:
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "PPO checkpoint(s) not found: " + ", ".join(missing[:3])
                + ("..." if len(missing) > 3 else "")
                + ". Train them with `make train` / `execshift train`, or run "
                "`make reproduce-paper-fast` for the reduced models in models_fast/.")
        from ..rl import PPOEnsemble  # torch / SB3 are imported only when RL is used

        self._impl = PPOEnsemble(list(paths), name=name)
        self.paths = [Path(p) for p in paths]
        self.name = name

    @classmethod
    def from_study(cls, models_dir: Path, impact_model: str, replicate: int,
                   single_member: bool = False,
                   ensemble_size: int = ENSEMBLE_SIZE) -> PPOPolicy:
        paths = member_paths(models_dir, impact_model, replicate, ensemble_size)
        if single_member:
            return cls(paths[:1], name="RL1")
        return cls(paths, name="RL")

    @property
    def ensemble(self):
        """The underlying frozen ``PPOEnsemble`` (used by the frozen gate features)."""
        return self._impl

    def reset(self, ex: Execution) -> None:
        return None

    def act(self, ex: Execution) -> np.ndarray:
        return self._impl(ex)

    def __call__(self, ex: Execution) -> np.ndarray:
        return self.act(ex)

    def member_actions(self, obs: np.ndarray) -> np.ndarray:
        return self._impl.member_actions(obs)

    def member_values(self, obs: np.ndarray) -> np.ndarray:
        return self._impl.member_values(obs)

    def parameters(self) -> dict:
        return {"checkpoints": [p.name for p in self.paths]}
