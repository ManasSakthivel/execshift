"""Statistical procedures (pre-registered in research/PREREGISTRATION.md, section 7).

Independent units.  Two sources of randomness are crossed: RL training replicates
(r = 1..R, independent seeds) and market episodes (e = 1..E_c per condition c, independent
scenario seeds).  Every policy is evaluated on the same episodes (common random numbers), so
comparisons are paired within episode.  Steps within an episode are never treated as units.

The primary interval is a two-way (crossed) bootstrap that resamples replicates and
episodes independently (Owen 2007, "The pigeonhole bootstrap"; Brennan, Harris & Hanes
1987), applied to the paired difference.  Conditions are combined with fixed, pre-registered
mixture weights; episodes are resampled within condition; the replicate resample is shared
across conditions because a replicate's policy is the same in every condition.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

B_BOOT = 10_000
BOOT_SEED = 20261001


def crossed_bootstrap(diffs: dict[str, np.ndarray], weights: dict[str, float],
                      B: int = B_BOOT, seed: int = BOOT_SEED,
                      resample_replicates: bool = True, normalize: bool = True) -> dict:
    """diffs[c] has shape (R, E_c). Statistic = sum_c w_c * mean(diffs[c]) (divided by
    sum_c w_c when `normalize`; signed contrasts such as difference-in-differences use
    normalize=False)."""
    rng = np.random.default_rng(seed)
    conds = [c for c in weights if weights[c] != 0]
    wsum = sum(weights[c] for c in conds) if normalize else 1.0
    R = diffs[conds[0]].shape[0]
    point = sum(weights[c] * diffs[c].mean() for c in conds) / wsum
    rep_counts = (rng.multinomial(R, np.full(R, 1 / R), size=B) if resample_replicates
                  else np.ones((B, R), dtype=int))  # (B, R)
    boot = np.zeros(B)
    for c in conds:
        D = diffs[c]
        E = D.shape[1]
        acc = np.empty((R, B))
        for s in range(0, B, 1000):
            n = min(1000, B - s)
            cnt = rng.multinomial(E, np.full(E, 1 / E), size=n).T.astype(np.float64)  # (E, n)
            acc[:, s:s + n] = D @ cnt / E
        rowmeans = acc.T  # (B, R)
        boot += weights[c] / wsum * (rowmeans * rep_counts).sum(axis=1) / R
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    # standardised effect: mean / sd of replicate-averaged per-episode differences
    pooled = np.concatenate([diffs[c].mean(axis=0) for c in conds])
    wts = np.concatenate([np.full(diffs[c].shape[1], abs(weights[c]) / diffs[c].shape[1])
                          for c in conds])
    wts = wts / wts.sum()
    sd = np.sqrt(np.sum(wts * (pooled - np.sum(wts * pooled)) ** 2))
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(p), "d_z": float(point / sd) if sd > 0 else float("nan"),
            "n_replicates": int(R), "n_episodes": {c: int(diffs[c].shape[1]) for c in conds},
            "B": int(B)}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        adj = min(1.0, max(running, (m - i) * pvals[k]))
        running = adj
        out[k] = {"p_raw": pvals[k], "p_holm": adj, "reject": adj < alpha}
    return out


def seed_level_wilcoxon(per_rep_diff: np.ndarray) -> dict:
    """Secondary: Wilcoxon signed-rank across the R replicate means (unit = replicate)."""
    res = stats.wilcoxon(per_rep_diff)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "n_negative": int((per_rep_diff < 0).sum()), "n": int(per_rep_diff.size)}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    y = label[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def risk_coverage(score: np.ndarray, regret: np.ndarray, n_points: int = 21):
    """Trust RL on the `coverage` fraction of episodes with the LOWEST score; fall back on the
    rest.  Returns coverage grid, mean regret among trusted episodes, and the mean selective
    regret over all episodes (fallback episodes contribute 0 regret by definition)."""
    order = np.argsort(score, kind="stable")
    cov = np.linspace(0, 1, n_points)
    sel, tot = [], []
    for c in cov:
        k = int(round(c * score.size))
        trusted = regret[order[:k]]
        sel.append(trusted.mean() if k else 0.0)
        tot.append(trusted.sum() / score.size)
    return cov, np.array(sel), np.array(tot)


def aurc(score: np.ndarray, regret: np.ndarray) -> float:
    """Area under the selective-regret (total) curve; lower = better ranking."""
    cov, _, tot = risk_coverage(score, regret, 101)
    return float(np.trapezoid(tot, cov))
