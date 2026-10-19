"""Typed extension points of the platform.

These are ``typing.Protocol`` classes (structural typing). Anything with the right methods
conforms, including the frozen study classes, which predate this module and cannot be edited.

Protocol -> implementations (consumers):

* MarketSimulator -> ``market.FrozenSimulator``, the pre-registered engine
  (evaluation engine, benchmarks)
* ImpactModel -> ``market.IMPACT_MODELS``: sqrt, linear, transient (scenarios, tests)
* LatencyModel -> ``market.LatencyModel`` (scenarios, benchmarks)
* ExecutionPolicy -> every class in ``agents``: TWAP, AC, ADAPT, PPO, blend, gated
  (evaluation engine)
* RLPolicy -> ``agents.PPOPolicy`` (uncertainty estimators)
* BaselinePolicy -> ``agents.TWAPPolicy`` / ``ACPolicy`` / ``ADAPTPolicy`` (manifests)
* UncertaintyEstimator -> ``uncertainty``: ensemble, OOD, shadow regret, RCG
  (gated policy, benchmarks)
* EvaluationMetric -> ``evaluation.metrics.METRICS`` (benchmarks, reports)
* Benchmark -> ``benchmarks.BenchmarkSuite`` (CLI)
* Experiment -> ``experiments.ExperimentConfig`` (benchmark runner, registry, CLI)
* ArtifactStore -> ``experiments.registry.FileArtifactStore`` (experiments, benchmarks, CLI)

The policy protocol is ``reset(ex)`` + ``act(ex)``. ``update``/``finalize`` hooks were
deliberately left out. Every policy in this project is frozen at evaluation time, and a hook
nothing calls would only be decoration. A future online-learning policy (see
docs/FUTURE_WORK.md, Tier 2) is the natural point to add one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .config import MarketParams
    from .sim import Execution, Market


@runtime_checkable
class MarketSimulator(Protocol):
    """Generates exogenous market paths and executes orders against them."""

    name: str

    def generate(self, params: MarketParams, rngs: Sequence[np.random.Generator]) -> Market:
        """One episode per generator; paths depend only on the generator (common random
        numbers across policies)."""
        ...

    def execution(self, market: Market) -> Execution:
        """A fresh execution state for a parent order on every episode of ``market``."""
        ...


@runtime_checkable
class ImpactModel(Protocol):
    """A price-impact model of the engine (cost of the trader's own trading)."""

    name: str
    temporary_exponent: float  # temporary cost per share ~ participation ** exponent
    permanent_exponent: float  # cumulative impact ~ (executed / volume) ** exponent

    def apply(self, params: MarketParams) -> MarketParams: ...


@runtime_checkable
class LatencyModel(Protocol):
    """Order-to-fill delay; the fill price moves along a Brownian bridge within the bin."""

    name: str
    latency_s: float

    def apply(self, params: MarketParams) -> MarketParams: ...


@runtime_checkable
class ExecutionPolicy(Protocol):
    """Maps execution state to the fraction of the parent order to sell in this bin.

    ``act`` returns an array of shape (B,), one value per episode in the batch. The engine
    clips it to remaining inventory and liquidates on the final step. A policy may read
    observable state only (``ex.obs()``, ``ex.q``, ``ex.t``). Reading latent simulator
    state (``ex.m.alpha``, ``ex.m.eta``) is a bug.
    """

    name: str

    def reset(self, ex: Execution) -> None: ...

    def act(self, ex: Execution) -> np.ndarray: ...


@runtime_checkable
class RLPolicy(ExecutionPolicy, Protocol):
    """A learned policy that can expose per-member actions (for epistemic uncertainty)."""

    def member_actions(self, obs: np.ndarray) -> np.ndarray:
        """Shape (K, B): actions of the K ensemble members in [-1, 1]."""
        ...


@runtime_checkable
class BaselinePolicy(ExecutionPolicy, Protocol):
    """A classical, hand-specified policy with (optionally tuned) parameters."""

    def parameters(self) -> dict[str, Any]: ...


@runtime_checkable
class UncertaintyEstimator(Protocol):
    """Scores each episode at order arrival from observable pre-trade data.

    Higher score means less trust in the RL policy. ``score`` returns shape (B,).
    """

    name: str

    def score(self, market: Market) -> np.ndarray: ...


@runtime_checkable
class EvaluationMetric(Protocol):
    """A scalar summary of a per-episode cost sample (or of a paired difference)."""

    name: str
    unit: str
    higher_is_better: bool

    def __call__(self, x: np.ndarray) -> float: ...


@runtime_checkable
class Benchmark(Protocol):
    """A declarative set of policies x scenarios x seeds x metrics, with a runner."""

    name: str
    category: str

    def run(self, **kwargs: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class Experiment(Protocol):
    """A fully specified, machine-readable experiment: what to run and how it is labelled.

    ``config_hash`` identifies the normalised configuration, so two runs with the same hash
    ran the same experiment. ``to_dict`` is what the run manifest stores.
    """

    name: str
    kind: str
    label: str

    @property
    def config_hash(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed-enough storage for experiment runs (manifests + files)."""

    def create_run(self, experiment_id: str) -> Path: ...

    def list_runs(self) -> list[str]: ...

    def load_manifest(self, experiment_id: str) -> dict[str, Any]: ...
