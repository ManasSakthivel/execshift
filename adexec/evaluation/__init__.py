"""Evaluation: policy-agnostic rollout engine, metrics engine, statistical layer."""

from .engine import RolloutResult, evaluate_policies, evaluate_scenario, rollout
from .metrics import METRICS, Metric
from .statistics import (
    PREREGISTERED_SESOI_BP,
    Comparison,
    crossed_comparison,
    paired_comparison,
    practical_verdict,
    unpaired_comparison,
)

__all__ = ["METRICS", "PREREGISTERED_SESOI_BP", "Comparison", "Metric", "RolloutResult",
           "crossed_comparison", "evaluate_policies", "evaluate_scenario",
           "paired_comparison", "practical_verdict", "rollout", "unpaired_comparison"]
