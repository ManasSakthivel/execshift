"""Simulator validation against empirical stylised facts (run before any policy result).

Each check returns a measured value, the target range taken from the empirical
literature, and PASS / FAIL.  Properties the simulator does not model by design (intraday
seasonality, leverage effect, long memory beyond the 30-minute horizon) are reported as
NOT MODELLED rather than silently omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from .config import BINS_PER_DAY, N_STEPS, NOMINAL, SPLIT_TAGS, MarketParams
from .policies import TWAP
from .sim import episode_rngs, generate_market, run_policy

LONG = 2000  # bins per long series (~2.6 trading days of 30 s bins)


@dataclass
class Check:
    name: str
    value: float
    lo: float
    hi: float
    source: str
    detail: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.lo <= self.value <= self.hi else "FAIL"


def _acf(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Mean over series of the per-series sample autocorrelation; x has shape (S, T)."""
    x = x - x.mean(axis=1, keepdims=True)
    den = (x**2).sum(axis=1)
    return np.array([((x[:, :-k] * x[:, k:]).sum(axis=1) / den).mean() for k in lags])


def _hill(x: np.ndarray, frac: float = 0.02) -> float:
    a = np.sort(np.abs(x))[::-1]
    k = max(int(frac * a.size), 10)
    return 1.0 / np.mean(np.log(a[:k] / a[k]))


def long_series(p: MarketParams = NOMINAL, n_series: int = 200, cond_idx: int = 0):
    return generate_market(p, episode_rngs(SPLIT_TAGS["simval"], cond_idx, np.arange(n_series)),
                           length=LONG)


def impact_curve(p: MarketParams, parts: np.ndarray, n: int = 3000, cond_idx: int = 1):
    """Mean impact cost of TWAP metaorders vs participation, in bp of arrival notional.

    Measured against a zero-impact, zero-spread counterfactual on the *same* episodes
    (common random numbers), so price noise cancels exactly and only impact remains.
    """
    out = []
    for j, pi in enumerate(parts):
        pp = replace(p, participation_lo=pi, participation_hi=pi)
        p0 = replace(pp, eta_mult=0.0, perm_coef=0.0, half_spread_mult=1e-12)
        cond = 100 + cond_idx * 50 + j
        m = generate_market(pp, episode_rngs(SPLIT_TAGS["simval"], cond, np.arange(n)))
        m0 = generate_market(p0, episode_rngs(SPLIT_TAGS["simval"], cond, np.arange(n)))
        IS, _ = run_policy(m, TWAP())
        IS0, _ = run_policy(m0, TWAP())
        spread_cost = 1e4 * np.mean(m.half_spread[:, -N_STEPS - 1:-1], axis=1)
        out.append(np.mean(IS - IS0 - spread_cost))
    return np.array(out)


def run_checks(p: MarketParams = NOMINAL) -> tuple[list[Check], dict]:
    m = long_series(p)
    r = m.ret[:, 1:] / m.sigma_nom[:, None]
    checks: list[Check] = []
    art: dict = {}

    kurt = stats.kurtosis(r, axis=1).mean()
    checks.append(Check("Heavy tails: excess kurtosis of 30 s returns", kurt, 3.0, 50.0,
                        "Cont (2001) SF-3; intraday excess kurtosis >> 0"))
    hill = np.median([_hill(row) for row in r])
    checks.append(Check("Heavy tails: Hill tail index of |r| (top 2%)", hill, 2.0, 5.0,
                        "Gopikrishnan et al. (1999) 'inverse cubic law', alpha ~ 3 (2-5)"))
    aggs = {}
    for agg in (1, 10, 60):
        T = (r.shape[1] // agg) * agg
        ra = r[:, :T].reshape(r.shape[0], -1, agg).sum(axis=2)
        aggs[agg] = float(stats.kurtosis(ra, axis=1).mean())
    art["kurtosis_by_agg"] = aggs
    checks.append(Check("Aggregational Gaussianity: kurtosis(30 min) / kurtosis(30 s)",
                        aggs[60] / aggs[1], 0.0, 0.5, "Cont (2001) SF-4"))
    lags = np.arange(1, 61)
    acf_r = _acf(r, lags)
    acf_abs = _acf(np.abs(r), lags)
    art["acf_r"], art["acf_abs"], art["lags"] = acf_r, acf_abs, lags
    checks.append(Check("No linear autocorrelation: max |ACF(r)| at lags 1-60",
                        np.abs(acf_r).max(), 0.0, 0.05, "Cont (2001) SF-1"))
    checks.append(Check("Volatility clustering: ACF(|r|) at lag 1", acf_abs[0], 0.05, 0.5,
                        "Cont (2001) SF-7"))
    checks.append(Check("Volatility clustering: ACF(|r|) at lag 20 (10 min)", acf_abs[19],
                        0.02, 0.4, "Cont (2001) SF-7, slow decay"))
    lv = np.log(m.volume[:, 1:])
    cvv = np.mean([np.corrcoef(lv[i], np.abs(r[i]))[0, 1] for i in range(len(r))])
    checks.append(Check("Volume-volatility correlation corr(log V, |r|)", cvv, 0.1, 0.7,
                        "Cont (2001) SF-10; Karpoff (1987)"))
    ls = np.log(m.half_spread)
    rv = np.log(m.sigma)
    csv = np.mean([np.corrcoef(ls[i], rv[i])[0, 1] for i in range(len(r))])
    checks.append(Check("Spread-volatility correlation corr(log s, log sigma)", csv, 0.1, 0.9,
                        "Bouchaud et al. (2018) ch. 12; Wyart et al. (2008)"))
    ld = np.log(m.depth_obs)
    cdv = np.mean([np.corrcoef(ld[i], rv[i])[0, 1] for i in range(len(r))])
    checks.append(Check("Depth-volatility correlation corr(log depth, log sigma)", cdv, -0.9,
                        -0.05, "Bouchaud et al. (2018); liquidity withdraws in volatile periods"))
    ac_depth = _acf(ld, np.array([1, 20]))
    checks.append(Check("Liquidity persistence: ACF(log depth) lag 1", ac_depth[0], 0.5, 0.99,
                        "Liquidity is persistent but mean-reverting (Bouchaud et al. 2018)"))
    cv_depth = float(np.mean(np.std(m.depth_obs, axis=1) / np.mean(m.depth_obs, axis=1)))
    checks.append(Check("Liquidity variation: coefficient of variation of depth", cv_depth, 0.2,
                        1.0, "Intraday depth CV of order 0.3-0.6 in large caps"))
    # predictability of the next return from the observable signal
    s = m.signal[:, :-1].ravel()
    rn = r.ravel()
    r2 = np.corrcoef(s, rn)[0, 1] ** 2
    checks.append(Check("Signal predictive R^2 for next 30 s return", r2, 0.005, 0.05,
                        "Cont, Kukanov & Stoikov (2014); Kolm et al. (2023): low single-digit %"))
    # square-root law of metaorder impact
    parts = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    curve = impact_curve(p, parts)
    slope = np.polyfit(np.log(parts), np.log(np.maximum(curve, 1e-6)), 1)[0]
    art["impact_parts"], art["impact_cost"] = parts, curve
    lo, hi = (0.4, 0.75) if p.impact_model != "linear" else (0.8, 1.1)
    checks.append(Check(f"Impact concavity: log-log slope of cost vs participation "
                        f"[{p.impact_model}]", slope, lo, hi,
                        "Almgren et al. (2005) 0.6; Toth et al. (2011) 0.5 (square-root law)"))
    # implied square-root-law prefactor Y: cost ~ (2/3) * Y * sigma_daily * sqrt(Q / ADV)
    j = int(np.argmin(np.abs(parts - 0.08)))
    q_adv = parts[j] * N_STEPS / BINS_PER_DAY
    sig_day = p.sigma_med * p.sigma_mult * np.sqrt(BINS_PER_DAY)
    y_hat = curve[j] / 1e4 / ((2 / 3) * sig_day * np.sqrt(q_adv))
    checks.append(Check("Impact level: implied square-root-law prefactor Y at 8% participation",
                        y_hat, 0.3, 1.5, "Toth et al. (2011); Bouchaud et al. (2018): Y ~ 0.5-1"))
    art["series"] = m
    return checks, art
