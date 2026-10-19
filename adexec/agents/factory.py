"""Build policies from declarative specs, so no evaluator contains policy-specific code.

A spec is a mapping with a ``kind`` and kind-specific fields::

    {"kind": "twap"}
    {"kind": "ac", "kappa": -1.0}           # or {"kind": "ac", "tuned": true}
    {"kind": "adapt", "weights": [...]}     # or {"kind": "adapt", "tuned": true}
    {"kind": "ppo", "replicate": 0}         # 5-member ensemble of the study's checkpoints
    {"kind": "ppo", "replicate": 0, "single_member": true}
    {"kind": "ppo", "models_dir": "runs/<id>/models", "ensemble_size": 2}   # a trained run
    {"kind": "blend", "w": 0.5, "first": {...}, "second": {...}}
    {"kind": "gated", "estimator": "ood", "quantile": 0.95,
     "primary": {...}, "fallback": {"kind": "twap"}}

``tuned`` reads the study's tuned values from ``baselines_<impact>.json``. They were tuned on
the train split, so using them on new benchmark streams is legitimate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths
from ..config import ENSEMBLE_SIZE
from .classical import ACPolicy, ADAPTPolicy, BlendPolicy, TWAPPolicy
from .gated import GatedPolicy
from .learned import PPOPolicy

POLICY_KINDS = ("twap", "ac", "adapt", "ppo", "blend", "gated")


@dataclass
class PolicyContext:
    """Everything a spec may need to resolve: impact model and artifact locations."""

    impact_model: str = "sqrt"
    models_dir: Path = field(default_factory=paths.models_dir)
    baselines_dir: Path = field(default_factory=paths.baselines_dir)

    def tuned_baselines(self) -> dict[str, Any]:
        p = Path(self.baselines_dir) / f"baselines_{self.impact_model}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `make tune` to tune the baselines")
        return json.loads(p.read_text())


def make_policy(spec: Mapping[str, Any], ctx: PolicyContext | None = None):
    ctx = ctx or PolicyContext()
    kind = str(spec.get("kind", "")).lower()
    name = spec.get("name")
    if kind == "twap":
        pol: Any = TWAPPolicy()
    elif kind == "ac":
        kappa = ctx.tuned_baselines()["AC"]["kappa"] if spec.get("tuned") else spec["kappa"]
        pol = ACPolicy(kappa)
    elif kind == "adapt":
        w = (ctx.tuned_baselines()["ADAPT"]["weights"] if spec.get("tuned")
             else spec["weights"])
        pol = ADAPTPolicy(w)
    elif kind == "ppo":
        pol = PPOPolicy.from_study(Path(spec.get("models_dir", ctx.models_dir)),
                                   spec.get("impact_model", ctx.impact_model),
                                   int(spec.get("replicate", 0)),
                                   bool(spec.get("single_member", False)),
                                   int(spec.get("ensemble_size", ENSEMBLE_SIZE)))
    elif kind == "blend":
        pol = BlendPolicy(make_policy(spec["first"], ctx), make_policy(spec["second"], ctx),
                          float(spec["w"]))
    elif kind == "gated":
        pol = _make_gated(spec, ctx)
    else:
        raise ValueError(f"unknown policy kind {kind!r}; expected one of {POLICY_KINDS}")
    if name:
        pol.name = str(name)
    return pol


def _make_gated(spec: Mapping[str, Any], ctx: PolicyContext) -> GatedPolicy:
    from ..uncertainty import make_estimator

    primary = make_policy(spec.get("primary", {"kind": "ppo", "replicate": 0}), ctx)
    fallback = make_policy(spec.get("fallback", {"kind": "twap"}), ctx)
    est = make_estimator(spec["estimator"], primary=primary, fallback=fallback,
                         impact_model=ctx.impact_model)
    if "threshold" in spec:
        thr = float(spec["threshold"])
    else:  # quantile of the estimator on in-distribution TRAIN episodes (as the study's gates)
        thr = est.train_quantile(float(spec.get("quantile", 0.95)))
    return GatedPolicy.from_estimator(primary, fallback, est, thr)


def describe_policy(pol) -> dict[str, Any]:
    """Manifest entry for a constructed policy."""
    params = pol.parameters() if hasattr(pol, "parameters") else {}
    return {"name": pol.name, "class": type(pol).__name__, "parameters": params}
