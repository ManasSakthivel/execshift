"""Classical policies behind the ``ExecutionPolicy`` interface.

Each class wraps the frozen implementation in ``adexec/policies.py`` and delegates to it,
so the sell quantities are bit-identical to the pre-registered study
(``tests/test_agents.py`` checks this).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import policies as _frozen
from ..sim import Execution


class _Delegate:
    """Adapter from the frozen ``policy(ex)`` callables to ``reset``/``act``."""

    name: str
    _impl: Any

    def reset(self, ex: Execution) -> None:
        if hasattr(self._impl, "reset"):
            self._impl.reset(ex)

    def act(self, ex: Execution) -> np.ndarray:
        return self._impl(ex)

    # the frozen engine (sim.rollout) calls policies directly
    def __call__(self, ex: Execution) -> np.ndarray:
        return self.act(ex)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"


class TWAPPolicy(_Delegate):
    """Sell the remaining inventory evenly over the remaining bins (the study's fallback)."""

    def __init__(self) -> None:
        self._impl = _frozen.TWAP()
        self.name = "TWAP"

    def parameters(self) -> dict[str, Any]:
        return {}


class ACPolicy(_Delegate):
    """Static Almgren-Chriss trajectory; ``kappa`` < 0 gives the mirrored back-loaded form."""

    def __init__(self, kappa: float) -> None:
        self._impl = _frozen.AlmgrenChriss(kappa)
        self.kappa = float(kappa)
        self.name = "AC"

    def parameters(self) -> dict[str, Any]:
        return {"kappa": self.kappa}


class ADAPTPolicy(_Delegate):
    """Tuned opportunistic heuristic over the same observable features as the RL agent."""

    WEIGHT_NAMES = ("signal", "depth", "spread", "rv", "volume")

    def __init__(self, weights) -> None:
        w = [float(x) for x in weights]
        if len(w) != 5:
            raise ValueError("ADAPT takes 5 weights (signal, depth, spread, rv, volume)")
        self._impl = _frozen.Adaptive(*w)
        self.weights = w
        self.name = "ADAPT"

    def parameters(self) -> dict[str, Any]:
        return {"weights": self.weights, "weight_names": list(self.WEIGHT_NAMES)}


class BlendPolicy(_Delegate):
    """Per-step convex combination ``w * first + (1 - w) * second`` (the study's A3 control)."""

    def __init__(self, first, second, w: float) -> None:
        self._impl = _frozen.Blend(first, second, w)
        self.w = float(w)
        self.name = f"BLEND_{w:g}"

    def parameters(self) -> dict[str, Any]:
        return {"w": self.w}
