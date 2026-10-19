"""Metrics engine: one documented, tested implementation of every metric.

Units and sign conventions
--------------------------
* Costs are **implementation shortfall (IS) in basis points of arrival notional**.
  Positive IS is a cost, so **lower is better**.
* A *paired difference* ``a - b`` is computed per episode on the same market (common
  random numbers). Negative means policy ``a`` is cheaper.
* *Regret* of trusting RL on an episode is ``IS_RL - IS_fallback``. Positive means RL was
  the wrong choice for that episode.

Relationship to the frozen study code
-------------------------------------
The pre-registered analysis (``adexec/analysis.py``, ``adexec/stats.py``) has its own
implementations. They are hash-locked and cannot be redirected here. This module is the
implementation for all *new* code. ``tests/test_metrics.py`` checks that it agrees
numerically with the frozen versions wherever both exist (CVaR, AUROC, AUPRC, Spearman,
risk-coverage), so the two cannot silently drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps

from .. import stats as _frozen_stats

# --------------------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------------------


def implementation_shortfall_bps(quantities: np.ndarray, prices: np.ndarray,
                                 arrival_mid: float | np.ndarray = 1.0) -> np.ndarray:
    """IS of a SELL order in bp: ``1e4 * (X*M0 - sum_t v_t P_t) / (X*M0)``.

    ``quantities`` (B, T) are fractions of the parent order (rows sum to 1); ``prices``
    (B, T) are fill prices. Interpretation: the cost of execution relative to selling
    everything instantly at the arrival mid. Unit: bp. Lower is better.
    """
    q = np.asarray(quantities, dtype=float)
    p = np.asarray(prices, dtype=float)
    m0 = np.asarray(arrival_mid, dtype=float)
    if q.shape != p.shape:
        raise ValueError("quantities and prices must have the same shape")
    proceeds = (q * p).sum(axis=-1)
    return 1e4 * (q.sum(axis=-1) * m0 - proceeds) / (q.sum(axis=-1) * m0)


def mean_cost(x: np.ndarray) -> float:
    """Expected cost (bp). The study's primary endpoint aggregates with this."""
    return float(np.mean(x))


def median_cost(x: np.ndarray) -> float:
    """Typical cost (bp). Robust to the heavy-tailed price noise in realised IS."""
    return float(np.median(x))


def std_cost(x: np.ndarray, ddof: int = 0) -> float:
    """Dispersion of cost (bp). ``ddof=0`` matches the study's ``describe`` tables."""
    return float(np.std(x, ddof=ddof))


def quantile_cost(x: np.ndarray, q: float) -> float:
    """``q``-quantile of cost (bp); ``q=0.95`` is the 95% value-at-risk of execution cost."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    return float(np.quantile(x, q))


def cvar(x: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional value-at-risk of cost (bp): the mean of costs at or above the
    ``alpha``-quantile, i.e. the expected cost in the worst ``1 - alpha`` of orders.

    Uses the same definition as the study's ``analysis.describe`` (empirical quantile from
    ``np.quantile``, tail inclusive). Higher CVaR means more tail risk.
    """
    x = np.asarray(x, dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(x[x >= np.quantile(x, alpha)].mean())


# --------------------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------------------


def paired_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-episode ``a - b`` (bp). Requires the same episodes in the same order."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal shape, got {a.shape} vs {b.shape}")
    return a - b


def regret(is_policy: np.ndarray, is_reference: np.ndarray) -> np.ndarray:
    """Per-episode counterfactual regret ``IS_policy - IS_reference`` (bp).
    Positive means the policy cost more than the reference on that very episode."""
    return paired_difference(is_policy, is_reference)


def win_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of paired episodes on which ``a`` is strictly cheaper than ``b``; ties count
    one half. 0.5 means no systematic advantage. This is a rate, not an effect size: a 60%
    win rate can coexist with a higher *mean* cost if the losses are large."""
    d = paired_difference(a, b)
    return float(np.mean((d < 0) + 0.5 * (d == 0)))


def relative_degradation(value: float, reference: float) -> float:
    """``(value - reference) / |reference|``. Example: RL's mean cost under a shift relative
    to its in-distribution cost. Dimensionless; > 0 means worse than the reference."""
    if reference == 0:
        return float("nan")
    return float((value - reference) / abs(reference))


def difference_in_differences(a_shift: np.ndarray, b_shift: np.ndarray, a_ref: np.ndarray,
                              b_ref: np.ndarray) -> float:
    """``mean(a - b | shift) - mean(a - b | reference)`` (bp): how much ``a``'s edge over
    ``b`` changes under a shift. This is the study's H2 degradation ``D_c`` with
    a = RL, b = TWAP. Positive means the edge eroded."""
    return float(paired_difference(a_shift, b_shift).mean()
                 - paired_difference(a_ref, b_ref).mean())


# --------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) with a Student-t interval. Assumes i.i.d. units. Do NOT pass
    per-step values or replicate-by-episode arrays flattened: use the crossed bootstrap
    (``evaluation.statistics``) when two random factors are crossed."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    h = float(_sps.t.ppf(0.5 + level / 2, n - 1) * se)
    return m, m - h, m + h


def bootstrap_ci(x: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean,
                 B: int = 2000, seed: int = 0, level: float = 0.95
                 ) -> tuple[float, float, float]:
    """(point, lo, hi): percentile bootstrap over i.i.d. units with a fixed seed."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (B, x.size))
    boots = np.array([statistic(x[i]) for i in idx])
    lo, hi = np.quantile(boots, [0.5 - level / 2, 0.5 + level / 2])
    return float(statistic(x)), float(lo), float(hi)


# --------------------------------------------------------------------------------------
# Association / discrimination (score vs regret)
# --------------------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation in [-1, 1]. For a gate score vs regret, > 0 means the score ranks
    harmful episodes higher. The study's H3 headline is rho = +0.108 (weak)."""
    return float(_sps.spearmanr(x, y)[0])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a random positive episode outscores a random negative one.
    0.5 = chance. The study's H3 headline: AUROC(regret > 0) = 0.542, barely above chance.
    Delegates to the frozen implementation so the two cannot disagree."""
    return _frozen_stats.auroc(np.asarray(score), np.asarray(label, dtype=bool))


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision. Compare with the base rate ``mean(label)``, not with 0.5."""
    return _frozen_stats.auprc(np.asarray(score), np.asarray(label, dtype=bool))


def risk_coverage(score: np.ndarray, regret_: np.ndarray, n_points: int = 21):
    """Selective-prediction curve. Trust RL on the lowest-score ``coverage`` fraction and
    fall back on the rest. Returns (coverage, mean regret of trusted, total regret / n)."""
    return _frozen_stats.risk_coverage(np.asarray(score), np.asarray(regret_), n_points)


# --------------------------------------------------------------------------------------
# Registry of sample metrics (used by benchmarks and reports)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A scalar summary of a per-episode cost sample (implements ``EvaluationMetric``)."""

    name: str
    unit: str
    higher_is_better: bool
    fn: Callable[[np.ndarray], float]
    description: str

    def __call__(self, x: np.ndarray) -> float:
        return float(self.fn(np.asarray(x, dtype=float)))


METRICS: dict[str, Metric] = {m.name: m for m in (
    Metric("mean", "bp", False, mean_cost, "expected implementation shortfall"),
    Metric("median", "bp", False, median_cost, "median implementation shortfall"),
    Metric("std", "bp", False, std_cost, "standard deviation of shortfall (ddof=0)"),
    Metric("p05", "bp", False, lambda x: quantile_cost(x, 0.05), "5th percentile of cost"),
    Metric("p95", "bp", False, lambda x: quantile_cost(x, 0.95),
           "95th percentile of cost (VaR95)"),
    Metric("cvar95", "bp", False, lambda x: cvar(x, 0.95),
           "mean cost of the worst 5% of orders (CVaR95)"),
    Metric("cvar99", "bp", False, lambda x: cvar(x, 0.99),
           "mean cost of the worst 1% of orders (CVaR99)"),
)}

__all__ = ["METRICS", "Metric", "auprc", "auroc", "bootstrap_ci", "cvar",
           "difference_in_differences", "implementation_shortfall_bps", "mean_ci",
           "mean_cost", "median_cost", "paired_difference", "pearson", "quantile_cost",
           "regret", "relative_degradation", "risk_coverage", "spearman", "std_cost",
           "win_rate"]
