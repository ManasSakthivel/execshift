"""Pre-trade gate features, computed at order arrival (t = 0) from observable data only.

The gate sees the 30-minute pre-trade window (same market, before the order arrives), the
order's own size, and the RL ensemble itself.  It never sees latent simulator state
(true alpha, true impact coefficient, regime multipliers).

Feature groups
  * market summary    -- realised vol, spread, depth, volume, signal-return correlation,
                         kurtosis, order size (all relative to the trader's nominal refs)
  * OOD distance      -- Mahalanobis distance of the market summary from the TRAIN-split
                         in-distribution summaries
  * policy uncertainty-- RL-ensemble action and value disagreement at the arrival state and
                         on pseudo-arrival states built from the window
  * shadow regret     -- replay of RL vs fallback over the pre-trade window using the
                         trader's *nominal* cost model and realised (observed) prices
  * shadow cost regret-- the same replay with prices frozen: the spread + impact cost
                         difference of the two schedules on the observed liquidity states
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from .config import N_STEPS, NOMINAL, WINDOW
from .sim import REF_PARTICIPATION, Market, build_obs, market_features, run_policy

MARKET_COLS = ("m_rv", "m_spread", "m_depth", "m_volume", "m_sigcorr", "m_kurt", "m_size",
               "m_absret")
OOD_COLS = MARKET_COLS[:7]
POLICY_COLS = ("u_ens_s0", "u_ens_win", "u_val_std", "u_val_mean")
FEATURE_COLS = (*MARKET_COLS, "ood_maha", *POLICY_COLS, "shadow_regret", "shadow_cost_regret")


def market_summary(m: Market) -> np.ndarray:
    f = market_features(m)
    w = slice(1, WINDOW + 1)  # indices observed before the order arrives at k = WINDOW
    r = m.ret[:, w] / m.sigma_nom[:, None]
    sig = m.signal[:, 0:WINDOW]  # signal at k predicts ret[k+1]
    sc = np.array([stats.pearsonr(sig[i], r[i])[0] for i in range(m.B)])
    return np.column_stack([
        f["log_rv"][:, WINDOW],
        f["log_spread"][:, w].mean(axis=1),
        f["log_depth"][:, w].mean(axis=1),
        f["log_volume"][:, WINDOW],
        sc,
        stats.kurtosis(r, axis=1),
        np.log(m.participation / REF_PARTICIPATION),
        np.log(np.mean(np.abs(r), axis=1)),
    ])


@dataclass
class OODModel:
    mean: np.ndarray
    prec: np.ndarray

    @classmethod
    def fit(cls, summ: np.ndarray) -> OODModel:
        x = summ[:, : len(OOD_COLS)]
        cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
        return cls(x.mean(axis=0), np.linalg.inv(cov))

    def score(self, summ: np.ndarray) -> np.ndarray:
        d = summ[:, : len(OOD_COLS)] - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.prec, d))


def shadow_market(m: Market, impact_model: str, freeze_prices: bool = False) -> Market:
    """Re-index the pre-trade window as a hypothetical execution using only observables.

    sigma is the trader's EWMA estimate, the impact coefficient is inferred from visible depth
    with the nominal prefactor, volume and spreads are as observed, alpha is not used.
    """
    f = market_features(m)
    L = m.ret.shape[1]

    def shift(a, pad):
        out = np.empty_like(a)
        out[:, WINDOW:] = a[:, : L - WINDOW]
        out[:, :WINDOW] = pad
        return out

    sig_hat = m.sigma_nom[:, None] * np.exp(f["log_rv"])
    sig_s = shift(sig_hat, sig_hat[:, :1])
    ret_s = shift(m.ret, 0.0) * (0.0 if freeze_prices else 1.0)
    eps_s = np.where(sig_s > 0, ret_s / np.roll(sig_s, 1, axis=1), 0.0)
    p = replace(NOMINAL, impact_model=impact_model, alpha_beta=0.0)
    return Market(
        p=p, sigma_nom=m.sigma_nom, hs_nom=m.hs_nom, eta_nom=m.eta_nom,
        participation=m.participation, sigma_bar=m.sigma_nom, eta_bar=m.eta_nom,
        vbar=np.full(m.B, p.vol_bin),
        sigma=sig_s, alpha=np.zeros_like(sig_s), signal=shift(m.signal, 0.0), ret=ret_s,
        eps=eps_s, half_spread=shift(m.half_spread, m.half_spread[:, :1]),
        eta=shift(m.eta_nom[:, None] / m.depth_obs, (m.eta_nom[:, None] / m.depth_obs)[:, :1]),
        depth_obs=shift(m.depth_obs, m.depth_obs[:, :1]),
        volume=shift(m.volume, m.volume[:, :1]),
        bridge_z=np.zeros((m.B, N_STEPS)),
    )


def policy_uncertainty(m: Market, ens) -> np.ndarray:
    f = market_features(m)
    ones = np.ones(m.B)
    o0 = build_obs(f, WINDOW, ones, 0, m.participation, REF_PARTICIPATION)
    a0 = ens.member_actions(o0)
    v0 = ens.member_values(o0)
    stds = []
    for k in range(WINDOW // 2, WINDOW + 1, 3):  # pseudo-arrival states over the window
        ok = build_obs(f, k, ones, 0, m.participation, REF_PARTICIPATION)
        stds.append(ens.member_actions(ok).std(axis=0))
    return np.column_stack([a0.std(axis=0), np.mean(stds, axis=0), v0.std(axis=0),
                            v0.mean(axis=0)])


def gate_features(m: Market, ens, fallback, ood: OODModel, impact_model: str) -> np.ndarray:
    summ = market_summary(m)
    sm = shadow_market(m, impact_model)
    is_rl, _ = run_policy(sm, ens)
    is_fb, _ = run_policy(sm, fallback)
    sc = shadow_market(m, impact_model, freeze_prices=True)
    c_rl, _ = run_policy(sc, ens)
    c_fb, _ = run_policy(sc, fallback)
    return np.column_stack([summ, ood.score(summ), policy_uncertainty(m, ens), is_rl - is_fb,
                            c_rl - c_fb])
