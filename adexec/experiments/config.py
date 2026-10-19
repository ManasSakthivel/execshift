"""Machine-readable experiment configuration (TOML) with a stable content hash.

One file describes one experiment: the market (condition, impact model, latency, regimes),
the policies, the model architecture and training budget (for training runs), and the
evaluation protocol (episodes, seeds, metrics, reference policy, practical threshold).

Example (``configs/examples/benchmark_liquidity.toml``)::

    [experiment]
    name = "liquidity-stress"
    kind = "evaluate"

    [market]
    conditions = ["ID", "LIQ-mod", "LIQ-sev"]
    impact_model = "sqrt"
    liquidity = "drought"          # named regime -> regime knobs (see REGIMES)

    [[policies]]
    kind = "twap"

    [[policies]]
    kind = "adapt"
    tuned = true

    [evaluation]
    n_episodes = 2000
    seeds = [0, 1]
    reference = "TWAP"

Rules
-----
* Unknown keys are errors, so a typo cannot silently change an experiment.
* ``config_hash`` is the SHA-256 of the canonical JSON of the *normalised* config (defaults
  filled in, keys sorted). Two files that mean the same thing get the same hash.
* Named regimes reuse the magnitudes of the frozen pre-registered shifts
  (``adexec/config.py``). They do not change those shifts: a config that sets a regime
  produces a scenario with ``is_preregistered == False``.
* Every config is labelled. The default label is ``POST-STUDY / EXPLORATORY``. The
  pre-registered study is described separately (``configs/study/preregistered.toml``) and
  is never run through this module.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import CONDITION_BY_NAME, SHIFTS
from ..market import IMPACT_MODELS, REGIME_KNOBS, Scenario

POST_STUDY_LABEL = "POST-STUDY / EXPLORATORY"
KINDS = ("evaluate", "benchmark", "train")


def _shift_params(name: str) -> dict[str, Any]:
    """Regime knobs of a frozen shift condition (whatever differs from nominal)."""
    from ..config import NOMINAL

    cond = CONDITION_BY_NAME[name].params
    return {k: getattr(cond, k) for k in sorted(REGIME_KNOBS)
            if getattr(cond, k) != getattr(NOMINAL, k)}


# Named regimes. Magnitudes are those of the frozen shift suite, so "stressed" volatility
# means exactly the VOL-mod multiplier. "normal" is the in-distribution value.
REGIMES: dict[str, dict[str, dict[str, Any]]] = {
    "volatility": {"normal": {}, "stressed": _shift_params("VOL-mod"),
                   "crisis": _shift_params("VOL-sev")},
    "liquidity": {"normal": {}, "thin": _shift_params("LIQ-mod"),
                  "drought": _shift_params("LIQ-sev")},
    "alpha": {"normal": {}, "decayed": _shift_params("ALPHA-mod"),
              "none": _shift_params("ALPHA-sev")},
    "order_size": {"normal": {}, "large": _shift_params("SIZE-mod"),
                   "very_large": _shift_params("SIZE-sev")},
}


class ConfigError(ValueError):
    """Raised for an invalid experiment configuration (message says how to fix it)."""


def _check_keys(section: str, d: Mapping[str, Any], allowed: set[str]) -> None:
    bad = sorted(set(d) - allowed)
    if bad:
        raise ConfigError(f"[{section}] has unknown key(s) {bad}; allowed: {sorted(allowed)}")


@dataclass(frozen=True)
class MarketConfig:
    conditions: tuple[str, ...] = ("ID",)
    impact_model: str = "sqrt"
    latency_s: float | None = None
    volatility: str = "normal"
    liquidity: str = "normal"
    alpha: str = "normal"
    order_size: str = "normal"
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for c in self.conditions:
            if c not in CONDITION_BY_NAME:
                raise ConfigError(f"unknown condition {c!r}; expected one of "
                                  f"{sorted(CONDITION_BY_NAME)}")
        if self.impact_model not in IMPACT_MODELS:
            raise ConfigError(f"unknown impact_model {self.impact_model!r}; expected one of "
                              f"{sorted(IMPACT_MODELS)}")
        for axis in REGIMES:
            v = getattr(self, axis)
            if v not in REGIMES[axis]:
                raise ConfigError(f"[market] {axis} = {v!r}; expected one of "
                                  f"{sorted(REGIMES[axis])}")
        bad = set(self.overrides) - REGIME_KNOBS
        if bad:
            raise ConfigError(f"[market.overrides] may only set {sorted(REGIME_KNOBS)}; "
                              f"got {sorted(bad)}")

    def regime_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for axis in REGIMES:
            out.update(REGIMES[axis][getattr(self, axis)])
        out.update(self.overrides)
        return out

    def scenarios(self) -> list[Scenario]:
        ov = self.regime_overrides()
        return [Scenario(c, self.impact_model, self.latency_s, ov) for c in self.conditions]


@dataclass(frozen=True)
class ModelConfig:
    """PPO architecture (defaults: the study's ``rl.PPO_KWARGS``)."""

    pi_layers: tuple[int, ...] = (64, 64)
    vf_layers: tuple[int, ...] = (64, 64)
    log_std_init: float = -1.0


@dataclass(frozen=True)
class TrainingConfig:
    """PPO training budget (defaults: the study's ``rl.PPO_KWARGS``, reduced steps)."""

    total_steps: int = 300_000
    n_envs: int = 64
    seeds: tuple[int, ...] = (0,)
    learning_rate: float = 1e-4
    n_steps: int = 120
    batch_size: int = 4096
    n_epochs: int = 5
    gamma: float = 1.0
    gae_lambda: float = 0.98
    clip_range: float = 0.2


@dataclass(frozen=True)
class EvaluationConfig:
    n_episodes: int = 1000
    seeds: tuple[int, ...] = (0,)
    metrics: tuple[str, ...] = ("mean", "median", "std", "p95", "cvar95")
    reference: str = "TWAP"
    bootstrap_B: int = 2000
    threshold_bp: float = 0.25
    threshold_source: str = "pre-registered SESOI (research/PREREGISTRATION.md section 3)"

    def __post_init__(self):
        from ..evaluation.metrics import METRICS

        if self.n_episodes < 2:
            raise ConfigError("[evaluation] n_episodes must be >= 2")
        bad = [m for m in self.metrics if m not in METRICS]
        if bad:
            raise ConfigError(f"[evaluation] unknown metric(s) {bad}; available: "
                              f"{sorted(METRICS)}")
        if self.threshold_bp != 0.25 and "pre-registered" in self.threshold_source:
            raise ConfigError("a threshold other than the pre-registered 0.25 bp needs its own "
                              "threshold_source (label it exploratory and say why)")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    kind: str = "evaluate"
    description: str = ""
    label: str = POST_STUDY_LABEL
    market: MarketConfig = field(default_factory=MarketConfig)
    policies: tuple[Mapping[str, Any], ...] = ({"kind": "twap"},)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    benchmark: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ConfigError(f"[experiment] kind = {self.kind!r}; expected one of {KINDS}")
        if not self.name or not all(ch.isalnum() or ch in "-_." for ch in self.name):
            raise ConfigError("[experiment] name must be non-empty: letters, digits, - _ .")
        if any(s in self.label.upper() for s in ("PRE-REGISTERED", "PREREGISTERED",
                                                   "CONFIRMATORY")):
            raise ConfigError("new experiments cannot be labelled pre-registered; the "
                              "pre-registered study is frozen (research/prereg.lock)")

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Normalised config in the same section layout as the TOML file, so
        ``config_from_dict(cfg.to_dict()) == cfg`` (stored in every run manifest)."""
        d = {"experiment": {"name": self.name, "kind": self.kind,
                            "description": self.description, "label": self.label},
             "market": asdict(self.market), "policies": [dict(p) for p in self.policies],
             "model": asdict(self.model), "training": asdict(self.training),
             "evaluation": asdict(self.evaluation), "benchmark": dict(self.benchmark)}
        return json.loads(json.dumps(d, sort_keys=True, default=list))

    @property
    def config_hash(self) -> str:
        return hash_config(self.to_dict())


def hash_config(d: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


_SECTIONS = {"experiment", "market", "policies", "model", "training", "evaluation",
             "benchmark"}


def _tuple_fields(d: dict, names: tuple[str, ...]) -> dict:
    return {k: (tuple(v) if k in names and isinstance(v, list) else v) for k, v in d.items()}


def config_from_dict(d: Mapping[str, Any]) -> ExperimentConfig:
    _check_keys("top level", d, _SECTIONS)
    exp = dict(d.get("experiment", {}))
    _check_keys("experiment", exp, {"name", "kind", "description", "label"})
    if "name" not in exp:
        raise ConfigError("[experiment] name is required")
    mk = dict(d.get("market", {}))
    _check_keys("market", mk, {f for f in MarketConfig.__dataclass_fields__})
    model = dict(d.get("model", {}))
    _check_keys("model", model, set(ModelConfig.__dataclass_fields__))
    tr = dict(d.get("training", {}))
    _check_keys("training", tr, set(TrainingConfig.__dataclass_fields__))
    ev = dict(d.get("evaluation", {}))
    _check_keys("evaluation", ev, set(EvaluationConfig.__dataclass_fields__))
    pols = d.get("policies", [{"kind": "twap"}])
    if not isinstance(pols, list) or not pols:
        raise ConfigError("[[policies]] must be a non-empty array of tables")
    for p in pols:
        if "kind" not in p:
            raise ConfigError(f"policy {p} has no 'kind'")
    try:
        return ExperimentConfig(
            **exp,
            market=MarketConfig(**_tuple_fields(mk, ("conditions",))),
            policies=tuple(dict(p) for p in pols),
            model=ModelConfig(**_tuple_fields(model, ("pi_layers", "vf_layers"))),
            training=TrainingConfig(**_tuple_fields(tr, ("seeds",))),
            evaluation=EvaluationConfig(**_tuple_fields(ev, ("seeds", "metrics"))),
            benchmark=dict(d.get("benchmark", {})),
        )
    except TypeError as e:  # wrong value types surface here
        raise ConfigError(str(e)) from e


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file {p} not found (examples: configs/examples/*.toml)")
    try:
        d = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p}: invalid TOML: {e}") from e
    return config_from_dict(d)


def preregistered_conditions() -> tuple[str, ...]:
    return ("ID", *(c.name for c in SHIFTS))


__all__ = ["ConfigError", "EvaluationConfig", "ExperimentConfig", "KINDS", "MarketConfig",
           "ModelConfig", "POST_STUDY_LABEL", "REGIMES", "TrainingConfig", "config_from_dict",
           "hash_config", "load_config", "preregistered_conditions"]
