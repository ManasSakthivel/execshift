"""Policy-agnostic evaluation engine.

The engine knows one thing about a policy: the ``ExecutionPolicy`` interface
(``reset(ex)`` and ``act(ex)``; a plain ``policy(ex)`` callable such as a frozen study class
also works). It has no policy names, no ``isinstance`` checks and no special cases. Policies
are built elsewhere (``agents.make_policy``) and passed in.

Every policy evaluated in one call sees the same generated market, so all comparisons are
paired by episode (common random numbers) and per-episode regret is exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..market import DEFAULT_SIMULATOR, Scenario, benchmark_rngs
from ..sim import Market


@dataclass
class RolloutResult:
    """Per-episode outcome of one policy on one batch of episodes."""

    policy: str
    is_bps: np.ndarray  # realised implementation shortfall (bp of arrival notional)
    is_cv_bps: np.ndarray  # control-variate IS: unbiased for E[IS], lower variance
    trades: np.ndarray  # (B, N_STEPS) fractions of the parent order sold per bin

    @property
    def n(self) -> int:
        return int(self.is_bps.size)


def rollout(market: Market, policy, simulator=DEFAULT_SIMULATOR,
            denoise: bool = False) -> RolloutResult:
    """Execute ``policy`` on every episode of ``market``.

    Numerically identical to the frozen ``sim.rollout`` (``tests/test_evaluation.py``).
    """
    ex = simulator.execution(market)
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset(ex)
    act = getattr(policy, "act", None) or policy
    while not ex.done:
        ex.step(act(ex), denoise=denoise)
    return RolloutResult(getattr(policy, "name", type(policy).__name__), ex.shortfall_bps(),
                         ex.shortfall_cv_bps(), ex.trades)


def evaluate_policies(market: Market, policies: Mapping[str, object],
                      simulator=DEFAULT_SIMULATOR) -> dict[str, RolloutResult]:
    """Roll out every policy on the same market (common random numbers)."""
    return {name: rollout(market, pol, simulator) for name, pol in policies.items()}


def evaluate_scenario(scenario: Scenario, policies: Mapping[str, object], n_episodes: int,
                      seed: int = 0, simulator=DEFAULT_SIMULATOR) -> dict[str, RolloutResult]:
    """Generate ``n_episodes`` of ``scenario`` from the post-study benchmark stream (tag 66)
    and evaluate every policy on them."""
    market = simulator.generate(scenario.params(), benchmark_rngs(scenario, n_episodes, seed))
    return evaluate_policies(market, policies, simulator)
