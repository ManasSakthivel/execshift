"""Market layer: simulator adapter, impact and latency models, scenarios.

The engine itself (``adexec/sim.py``) and the 12 pre-registered shift conditions
(``adexec/config.py::SHIFTS``) are frozen by the pre-registration lock. This package does not
re-implement them. It gives them typed, named, configurable handles:

* ``FrozenSimulator``: the pre-registered engine behind the ``MarketSimulator`` protocol.
* ``IMPACT_MODELS``: the three impact models the engine implements, with their declared
  scaling exponents. ``tests/test_market.py`` checks the engine against these exponents.
* ``LatencyModel``: a named order-to-fill delay.
* ``Scenario``: (condition, impact model, latency, extra overrides) -> ``MarketParams``.
  It reproduces the study's ``condition_params`` rule exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..config import CONDITION_BY_NAME, CONDITIONS, NOMINAL, SHIFTS, MarketParams
from ..sim import Execution, Market, episode_rngs, generate_market

# --------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------


class FrozenSimulator:
    """The pre-registered synthetic market engine (``adexec/sim.py``), unchanged."""

    name = "adexec-sim-v1"

    def generate(self, params: MarketParams, rngs: Sequence[np.random.Generator]) -> Market:
        return generate_market(params, list(rngs))

    def execution(self, market: Market) -> Execution:
        return Execution(market)


DEFAULT_SIMULATOR = FrozenSimulator()

# --------------------------------------------------------------------------------------
# Impact models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactModelSpec:
    """One of the engine's impact models.

    The engine selects behaviour via ``MarketParams.impact_model``. The exponents below are
    the *declared* scaling of the engine's cost terms. They describe the code; they do not
    drive it. All three models are matched in cost at 7.5% participation
    (``impact_ref_participation``).
    """

    name: str
    description: str
    temporary_exponent: float
    permanent_exponent: float
    transient_share: float = 0.0

    def apply(self, params: MarketParams) -> MarketParams:
        return replace(params, impact_model=self.name)


IMPACT_MODELS: dict[str, ImpactModelSpec] = {
    "sqrt": ImpactModelSpec(
        "sqrt", "square-root law: concave temporary and cumulative impact "
        "(Toth et al. 2011; primary model)", 0.5, 0.5),
    "linear": ImpactModelSpec(
        "linear", "Almgren-Chriss linear temporary and permanent impact, matched to sqrt at "
        "the reference participation", 1.0, 1.0),
    "transient": ImpactModelSpec(
        "transient", "square-root law with half of the temporary impact decaying "
        "exponentially (tau = 120 s; Obizhaeva-Wang / propagator-type)", 0.5, 0.5, 0.5),
}

# --------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyModel:
    """A fixed order-to-fill delay. Fills are priced on a Brownian bridge inside the bin."""

    name: str
    latency_s: float

    def apply(self, params: MarketParams) -> MarketParams:
        if self.latency_s < 0:
            raise ValueError("latency must be non-negative")
        return replace(params, latency_s=float(self.latency_s))


NOMINAL_LATENCY = LatencyModel("nominal", NOMINAL.latency_s)
# the pre-registered H6 latency robustness grid (scripts/run_confirmatory.py)
STUDY_LATENCY_GRID_S: tuple[float, ...] = (0.0, 1.0, 5.0)

# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------

_PARAM_FIELDS = {f.name for f in dataclasses.fields(MarketParams)}
# regime knobs a scenario may override (everything else is part of the frozen market model)
REGIME_KNOBS = frozenset({
    "sigma_mult", "half_spread_mult", "eta_mult", "depth_obs_mult", "volume_mult",
    "alpha_beta_mult", "participation_mult", "latency_s", "impact_model",
})


@dataclass(frozen=True)
class Scenario:
    """A market scenario: a named base condition plus impact model, latency and overrides.

    ``condition`` must be one of the 13 pre-registered conditions (ID + 12 shifts), whose
    definitions are taken from ``adexec/config.py`` unchanged. ``overrides`` (regime knobs
    only) create a NEW scenario that is not part of the pre-registered suite;
    ``is_preregistered`` tells the two apart.

    Latency rule (identical to the study's ``condition_params``): an explicit ``latency_s``
    replaces the nominal latency **except** in LATENCY-family conditions, whose latency is
    the shift itself.
    """

    condition: str = "ID"
    impact_model: str = "sqrt"
    latency_s: float | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self):
        if self.condition not in CONDITION_BY_NAME:
            raise ValueError(f"unknown condition {self.condition!r}; "
                             f"expected one of {sorted(CONDITION_BY_NAME)}")
        if self.impact_model not in IMPACT_MODELS:
            raise ValueError(f"unknown impact model {self.impact_model!r}")
        bad = set(self.overrides) - REGIME_KNOBS
        if bad:
            raise ValueError(f"overrides may only set regime knobs {sorted(REGIME_KNOBS)}; "
                             f"got {sorted(bad)}")
        object.__setattr__(self, "overrides", dict(sorted(self.overrides.items())))

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        s = f"{self.condition}/{self.impact_model}"
        if self.latency_s is not None:
            s += f"/lat{self.latency_s:g}"
        if self.overrides:
            s += "/" + ",".join(f"{k}={v}" for k, v in self.overrides.items())
        return s

    @property
    def family(self) -> str:
        return CONDITION_BY_NAME[self.condition].family

    @property
    def is_preregistered(self) -> bool:
        return not self.overrides

    def params(self) -> MarketParams:
        p = IMPACT_MODELS[self.impact_model].apply(CONDITION_BY_NAME[self.condition].params)
        if self.latency_s is not None and self.family != "LATENCY":
            p = LatencyModel("override", self.latency_s).apply(p)
        if self.overrides:
            p = replace(p, **self.overrides)
        return p

    def to_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "impact_model": self.impact_model,
                "latency_s": self.latency_s, "overrides": dict(self.overrides),
                "label": self.label}

    def fingerprint(self) -> str:
        """Hash of the *resolved* market parameters (what the engine will actually see)."""
        d = dataclasses.asdict(self.params())
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def preregistered_scenarios(impact_model: str = "sqrt",
                            latency_s: float | None = None) -> list[Scenario]:
    """The 13 pre-registered conditions (ID + the 12 frozen shifts) as scenarios."""
    return [Scenario(c.name, impact_model, latency_s) for c in CONDITIONS]


def shift_family_scenarios(family: str, impact_model: str = "sqrt") -> list[Scenario]:
    return [Scenario(c.name, impact_model) for c in SHIFTS if c.family == family]


# --------------------------------------------------------------------------------------
# Seed streams
# --------------------------------------------------------------------------------------

# Split tags 11/22/33/44/55 belong to the pre-registered study (config.SPLIT_TAGS). New,
# post-study benchmarks draw from tag 66 so that they never reuse a study episode and can
# never touch the confirmatory test stream (tag 33).
BENCHMARK_STREAM_TAG = 66


def benchmark_rngs(scenario: Scenario, n: int, seed: int = 0) -> list[np.random.Generator]:
    """Episode generators for post-study benchmarks: SeedSequence([66, cond, seed, i]).

    Episodes are keyed by condition index and benchmark seed, not by impact model or
    latency. Every impact/latency variant of a condition therefore sees the same exogenous
    randomness (common random numbers across robustness settings).
    """
    cond_idx = [c.name for c in CONDITIONS].index(scenario.condition)
    return [np.random.default_rng(np.random.SeedSequence([BENCHMARK_STREAM_TAG, cond_idx,
                                                          int(seed), int(i)]))
            for i in range(n)]


__all__ = [
    "BENCHMARK_STREAM_TAG", "DEFAULT_SIMULATOR", "FrozenSimulator", "IMPACT_MODELS",
    "ImpactModelSpec", "LatencyModel", "NOMINAL_LATENCY", "REGIME_KNOBS",
    "STUDY_LATENCY_GRID_S", "Scenario", "benchmark_rngs", "episode_rngs",
    "preregistered_scenarios", "shift_family_scenarios",
]
