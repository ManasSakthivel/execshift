"""Statistical layer: paired and unpaired comparisons that always report a practical effect.

Rules this module enforces
--------------------------
1. **Paired vs unpaired is explicit.** ``paired_comparison`` requires the same episodes in
   the same order (common random numbers), and its unit is the episode.
   ``unpaired_comparison`` is for independent samples. ``crossed_comparison`` is for the
   study's design: R training replicates crossed with E episodes.
2. **No significance without effect size.** Every result carries the point estimate in bp,
   a 95% CI, a standardised effect size and a verdict against a *practical threshold*
   (smallest effect size of interest, SESOI).
3. **Thresholds are labelled.** The pre-registered SESOI is 0.25 bp (``analysis.SESOI``,
   hash-locked). A different threshold must be passed explicitly with
   ``threshold_source``, and the verdict records it.

Bootstrap methodology
---------------------
* Paired, one replicate: percentile bootstrap over episodes, B resamples, fixed seed.
* Crossed (replicates x episodes): delegated to the frozen ``stats.crossed_bootstrap``
  (two-way "pigeonhole" bootstrap; Owen 2007), so new analyses use the same procedure as the
  confirmatory study.
* Two-sided p-value: twice the smaller bootstrap tail mass at 0, as in the study. The
  bootstrap resolution limits it to multiples of 1/B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import stats as _frozen_stats
from ..analysis import SESOI as PREREGISTERED_SESOI_BP

PREREGISTERED = "pre-registered SESOI (research/PREREGISTRATION.md section 3)"


@dataclass(frozen=True)
class Comparison:
    """Result of comparing policy ``a`` with policy ``b`` on cost (lower is better)."""

    a: str
    b: str
    design: str  # "paired" | "unpaired" | "crossed"
    estimate: float  # mean(a - b), bp; < 0 favours a
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float  # d_z (paired) or Cohen's d (unpaired)
    n: int
    threshold_bp: float
    threshold_source: str
    B: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    @property
    def verdict(self) -> str:
        return practical_verdict(self.estimate, self.ci_low, self.ci_high, self.threshold_bp)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(significant=self.significant, verdict=self.verdict)
        return d


def practical_verdict(estimate: float, ci_low: float, ci_high: float,
                      threshold_bp: float) -> str:
    """Classify a cost difference ``a - b`` (negative favours ``a``) against a threshold.

    * ``a practically better``: CI below 0 and the estimate beyond -threshold.
    * ``b practically better``: the mirror image.
    * ``significant but below threshold``: CI excludes 0 but the |estimate| is below the
      threshold. The study's H4 lands here: -0.013 bp [-0.024, -0.004] against 0.25 bp.
    * ``practically equivalent``: CI inside (-threshold, +threshold) and includes 0.
    * ``inconclusive``: none of the above (CI too wide to decide).
    """
    sig = ci_low > 0 or ci_high < 0
    if sig and ci_high < 0 and estimate < -threshold_bp:
        return "a practically better"
    if sig and ci_low > 0 and estimate > threshold_bp:
        return "b practically better"
    if sig:
        return "significant but below threshold"
    if -threshold_bp < ci_low and ci_high < threshold_bp:
        return "practically equivalent"
    return "inconclusive"


def _p_two_sided(boot: np.ndarray) -> float:
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


def paired_comparison(a: np.ndarray, b: np.ndarray, names: tuple[str, str] = ("a", "b"),
                      threshold_bp: float = PREREGISTERED_SESOI_BP,
                      threshold_source: str = PREREGISTERED, B: int = 10_000,
                      seed: int = 0) -> Comparison:
    """Paired percentile bootstrap of mean(a - b) over episodes (unit = episode)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.ndim != 1:
        raise ValueError("paired_comparison takes 1-D per-episode arrays; use "
                         "crossed_comparison for replicate x episode arrays")
    rng = np.random.default_rng(seed)
    n = d.size
    boot = np.empty(B)
    for s in range(0, B, 1000):  # chunked to bound memory at n x 1000
        k = min(1000, B - s)
        boot[s:s + k] = d[rng.integers(0, n, (k, n))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    sd = d.std(ddof=1)
    return Comparison(names[0], names[1], "paired", float(d.mean()), float(lo), float(hi),
                      _p_two_sided(boot), float(d.mean() / sd) if sd > 0 else float("nan"),
                      int(n), float(threshold_bp), threshold_source, int(B))


def unpaired_comparison(a: np.ndarray, b: np.ndarray, names: tuple[str, str] = ("a", "b"),
                        threshold_bp: float = PREREGISTERED_SESOI_BP,
                        threshold_source: str = PREREGISTERED, B: int = 10_000,
                        seed: int = 0) -> Comparison:
    """Independent-samples bootstrap of mean(a) - mean(b). Use only when the samples do
    NOT share episodes. With common random numbers, the paired test is far more powerful."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([a[rng.integers(0, a.size, a.size)].mean()
                     - b[rng.integers(0, b.size, b.size)].mean() for _ in range(B)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    est = float(a.mean() - b.mean())
    return Comparison(names[0], names[1], "unpaired", est, float(lo), float(hi),
                      _p_two_sided(boot), est / pooled if pooled > 0 else float("nan"),
                      int(a.size + b.size), float(threshold_bp), threshold_source, int(B))


def crossed_comparison(diffs: dict[str, np.ndarray], weights: dict[str, float],
                       names: tuple[str, str] = ("a", "b"),
                       threshold_bp: float = PREREGISTERED_SESOI_BP,
                       threshold_source: str = PREREGISTERED, B: int = 10_000,
                       seed: int = _frozen_stats.BOOT_SEED) -> Comparison:
    """Replicate x episode design (``diffs[c]`` has shape (R, E_c)); the frozen two-way
    crossed bootstrap with mixture weights over conditions."""
    r = _frozen_stats.crossed_bootstrap(diffs, weights, B=B, seed=seed)
    n = int(sum(r["n_episodes"].values()))
    return Comparison(names[0], names[1], "crossed", r["estimate"], r["ci_low"], r["ci_high"],
                      r["p_value"], r["d_z"], n, float(threshold_bp), threshold_source, int(B))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down correction within one family (the frozen study implementation)."""
    return _frozen_stats.holm(pvals, alpha)


def verdict_for_study_result(res: dict, threshold_bp: float = PREREGISTERED_SESOI_BP) -> str:
    """Apply ``practical_verdict`` to a comparison dict from ``results/authoritative.json``."""
    return practical_verdict(res["estimate"], res["ci_low"], res["ci_high"], threshold_bp)


__all__ = ["Comparison", "PREREGISTERED_SESOI_BP", "crossed_comparison", "holm",
           "paired_comparison", "practical_verdict", "unpaired_comparison",
           "verdict_for_study_result"]
