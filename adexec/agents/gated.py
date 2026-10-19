"""The uncertainty-gated policy as a real, executable ``ExecutionPolicy``.

In the pre-registered study the gate is never rolled out. Because it decides once per
episode at arrival, its cost is computed analytically as
``where(fallback, IS_fallback, IS_rl)`` (``gates.apply_gates``). This class rolls the gate
out step by step. ``tests/test_agents.py`` checks that the rollout gives exactly the
analytical cost, which turns the study's undocumented shortcut into a tested property.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..sim import Execution, Market


class GatedPolicy:
    """Decide once at arrival: TRUST the primary policy or FALL BACK, per episode.

    ``decide(market) -> bool array (B,)``: True means fall back. It may use only data
    observable at arrival (the pre-trade window, the order size, the policies themselves).
    ``from_estimator`` builds ``decide`` as ``estimator.score(market) > threshold``.
    """

    def __init__(self, primary, fallback, decide: Callable[[Market], np.ndarray],
                 name: str = "GATED") -> None:
        self.primary, self.fallback, self.decide = primary, fallback, decide
        self.name = name
        self.fallback_mask: np.ndarray | None = None

    @classmethod
    def from_estimator(cls, primary, fallback, estimator, threshold: float,
                       name: str | None = None) -> GatedPolicy:
        def decide(m: Market) -> np.ndarray:
            return np.asarray(estimator.score(m)) > threshold

        return cls(primary, fallback, decide, name or f"GATE[{estimator.name}>{threshold:g}]")

    def reset(self, ex: Execution) -> None:
        for p in (self.primary, self.fallback):
            if hasattr(p, "reset"):
                p.reset(ex)
        mask = np.asarray(self.decide(ex.m), dtype=bool)
        if mask.shape != (ex.m.B,):
            raise ValueError(f"decide() must return shape ({ex.m.B},), got {mask.shape}")
        self.fallback_mask = mask

    def act(self, ex: Execution) -> np.ndarray:
        if self.fallback_mask is None:
            raise RuntimeError("GatedPolicy.act called before reset")
        return np.where(self.fallback_mask, self.fallback(ex), self.primary(ex))

    def __call__(self, ex: Execution) -> np.ndarray:
        return self.act(ex)

    @property
    def fallback_rate(self) -> float:
        if self.fallback_mask is None:
            return float("nan")
        return float(np.mean(self.fallback_mask))

    def parameters(self) -> dict:
        return {"primary": getattr(self.primary, "name", "?"),
                "fallback": getattr(self.fallback, "name", "?")}
