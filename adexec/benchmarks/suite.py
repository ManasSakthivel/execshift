"""Declarative benchmark definition.

A ``BenchmarkSuite`` is data: which policies (as factory specs), on which scenarios, with
how many episodes and seeds, scored by which metrics, compared against which reference,
with which statistical procedure. ``runner.run_suite`` executes it. The suite contains no
policy code and the runner contains no policy names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..evaluation.metrics import METRICS
from ..market import Scenario

DEFAULT_METRICS = ("mean", "median", "std", "p95", "cvar95")
CATEGORIES = ("baseline", "in-distribution", "distribution-shift", "impact-robustness",
              "latency-robustness", "liquidity-robustness", "risk", "gate", "composite",
              "smoke", "custom")


def default_policy_name(spec: Mapping[str, Any]) -> str:
    if spec.get("name"):
        return str(spec["name"])
    kind = str(spec["kind"]).lower()
    if kind == "ppo":
        base = "RL1" if spec.get("single_member") else "RL"
        r = spec.get("replicate", 0)
        return base if r == 0 else f"{base}_r{int(r):02d}"
    if kind == "blend":
        return f"BLEND_{float(spec['w']):g}"
    if kind == "gated":
        return f"GATE_{spec['estimator']}"
    return {"twap": "TWAP", "ac": "AC", "adapt": "ADAPT"}.get(kind, kind.upper())


def named_policies(specs) -> dict[str, dict[str, Any]]:
    """``[spec, ...]`` -> ``{name: spec}``; duplicate names are an error."""
    out: dict[str, dict[str, Any]] = {}
    for s in specs:
        n = default_policy_name(s)
        if n in out:
            raise ValueError(f"two policies are both named {n!r}; give one a 'name'")
        out[n] = dict(s)
    return out


def spec_needs_checkpoints(spec: Mapping[str, Any]) -> bool:
    kind = str(spec.get("kind", "")).lower()
    if kind == "ppo":
        return True
    if kind == "blend":
        return spec_needs_checkpoints(spec["first"]) or spec_needs_checkpoints(spec["second"])
    if kind == "gated":
        return (spec_needs_checkpoints(spec.get("primary", {"kind": "ppo"}))
                or spec_needs_checkpoints(spec.get("fallback", {"kind": "twap"})))
    return False


@dataclass(frozen=True)
class StatisticalProcedure:
    """How comparisons are made. Defaults follow the study's conventions."""

    design: str = "paired"  # every policy sees the same episodes (common random numbers)
    estimator: str = "IScv"  # control-variate IS for means/differences; realised IS for risk
    bootstrap_B: int = 2000
    bootstrap_seed: int = 0
    threshold_bp: float = 0.25
    threshold_source: str = "pre-registered SESOI (research/PREREGISTRATION.md section 3)"
    unit: str = ("episode (one RL training replicate unless several are listed as separate "
                 "policies: training-seed variance is NOT in these intervals)")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class BenchmarkSuite:
    """Implements ``interfaces.Benchmark``."""

    name: str
    category: str
    description: str
    policies: Mapping[str, Mapping[str, Any]]
    scenarios: tuple[Scenario, ...]
    seeds: tuple[int, ...] = (0,)
    n_episodes: int = 1000
    metrics: tuple[str, ...] = DEFAULT_METRICS
    reference: str = "TWAP"
    statistics: StatisticalProcedure = field(default_factory=StatisticalProcedure)
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"category {self.category!r} not in {CATEGORIES}")
        if self.reference not in self.policies:
            raise ValueError(f"reference policy {self.reference!r} is not in the suite")
        bad = [m for m in self.metrics if m not in METRICS]
        if bad:
            raise ValueError(f"unknown metrics {bad}")
        if not self.scenarios:
            raise ValueError("a suite needs at least one scenario")
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique within a suite")

    @property
    def requires_checkpoints(self) -> bool:
        return any(spec_needs_checkpoints(s) for s in self.policies.values())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "category": self.category,
                "description": self.description,
                "policies": {k: dict(v) for k, v in self.policies.items()},
                "scenarios": [s.to_dict() | {"name": s.name,
                                             "is_preregistered": s.is_preregistered,
                                             "fingerprint": s.fingerprint()}
                              for s in self.scenarios],
                "seeds": list(self.seeds), "n_episodes": self.n_episodes,
                "metrics": list(self.metrics), "reference": self.reference,
                "statistics": self.statistics.to_dict(), "notes": list(self.notes)}

    def run(self, **kwargs: Any):
        from .runner import run_suite

        return run_suite(self, **kwargs)


__all__ = ["BenchmarkSuite", "CATEGORIES", "DEFAULT_METRICS", "StatisticalProcedure",
           "default_policy_name", "named_policies", "spec_needs_checkpoints"]
