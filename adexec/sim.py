"""Vectorised synthetic market and execution engine for a sell-side parent order.

The market is split into an *exogenous* part (unaffected fundamental price, volatility,
alpha, spread, liquidity, volume), generated once per episode from its seed, and an
*endogenous* part (the trader's own permanent / transient / temporary impact), computed
by the engine from the trader's actions.  Because the exogenous part depends only on the
episode seed, every policy evaluated on the same episode faces exactly the same market
(common random numbers), which makes paired comparisons and counterfactual regret exact.

Units: prices are relative to the arrival mid (M_0 = 1); costs are reported in basis
points of arrival notional.  Implementation shortfall for a sell order is
    IS = 1e4 * (X * M_0 - sum_t v_t P_t) / (X * M_0)        (lower is better).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .config import DT_SECONDS, N_STEPS, WINDOW, MarketParams


@dataclass
class Market:
    """Exogenous paths for a batch of B episodes.  Time index k = 0..W+N, decision t = k-W."""

    p: MarketParams
    # per-episode nominal characteristics (known to the trader) and realised values
    sigma_nom: np.ndarray
    hs_nom: np.ndarray
    eta_nom: np.ndarray
    participation: np.ndarray  # realised order size / nominal horizon volume
    sigma_bar: np.ndarray
    eta_bar: np.ndarray
    vbar: np.ndarray  # realised mean volume per bin
    # paths, shape (B, W+N+1)
    sigma: np.ndarray  # per-bin vol of the return from k to k+1
    alpha: np.ndarray
    signal: np.ndarray
    ret: np.ndarray  # ret[:, k] = unaffected return from k-1 to k (ret[:, 0] = 0)
    eps: np.ndarray  # standardised idiosyncratic innovation of ret[:, k]
    half_spread: np.ndarray
    eta: np.ndarray  # realised temporary-impact coefficient
    depth_obs: np.ndarray  # observable depth proxy (relative to nominal)
    volume: np.ndarray
    bridge_z: np.ndarray  # (B, N) Gaussian draws for intra-bin latency bridge

    @property
    def B(self) -> int:
        return self.sigma.shape[0]


def episode_rngs(split_tag: int, cond_idx: int, idx: np.ndarray) -> list[np.random.Generator]:
    return [np.random.default_rng(np.random.SeedSequence([split_tag, cond_idx, int(i)]))
            for i in idx]


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Stationary AR(1) with marginal sd `sd`."""
    innov = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n)
    innov[0] = rng.normal(0.0, sd)
    return lfilter([1.0], [1.0, -phi], innov)


def _student_t_std(rng: np.random.Generator, dof: float, n: int) -> np.ndarray:
    return rng.standard_t(dof, n) / np.sqrt(dof / (dof - 2.0))


def generate_market(p: MarketParams, rngs: list[np.random.Generator],
                    length: int | None = None) -> Market:
    """Draw exogenous paths for one episode per generator (draw order is fixed per episode)."""
    B = len(rngs)
    L = WINDOW + N_STEPS + 1 if length is None else int(length)
    out = {k: np.empty((B, L)) for k in
           ("sigma", "alpha", "signal", "ret", "eps", "half_spread", "eta", "depth_obs",
            "volume")}
    bridge = np.empty((B, N_STEPS))
    nom = np.empty((B, 4))
    phi_a = np.exp(-DT_SECONDS / p.alpha_tau_s)
    beta = p.alpha_beta * p.alpha_beta_mult
    eabs = np.sqrt((p.t_dof - 2) / np.pi) * np.exp(
        _lgamma((p.t_dof - 1) / 2) - _lgamma(p.t_dof / 2))  # E|standardised t|
    for b, rng in enumerate(rngs):
        # nominal stock characteristics (cross-sectional heterogeneity)
        sig_nom = p.sigma_med * np.exp(rng.normal(0, p.sigma_logsd))
        hs_nom = p.half_spread_med * np.exp(rng.normal(0, p.half_spread_logsd))
        eta_nom = p.eta_med * np.exp(rng.normal(0, p.eta_logsd))
        part = rng.uniform(p.participation_lo, p.participation_hi)
        nom[b] = (sig_nom, hs_nom, eta_nom, part)
        h = _ar1(rng, L, p.h_phi, p.h_sd)
        sigma = sig_nom * p.sigma_mult * np.exp(h - 0.5 * p.h_sd**2)
        alpha = _ar1(rng, L, phi_a, 1.0)
        noise = rng.normal(0, 1, L)
        signal = p.signal_rho * alpha + np.sqrt(1 - p.signal_rho**2) * noise
        eps = _student_t_std(rng, p.t_dof, L)
        ret = np.zeros(L)
        ret[1:] = sigma[:-1] * (beta * alpha[:-1] + np.sqrt(1 - beta**2) * eps[1:])
        eps[0] = 0.0
        u = _ar1(rng, L, p.spread_phi, p.spread_sd)
        hs = hs_nom * p.half_spread_mult * np.exp(
            p.spread_vol_load * h + u - 0.5 * (p.spread_vol_load**2 * p.h_sd**2 + p.spread_sd**2))
        w = _ar1(rng, L, p.liq_phi, p.liq_sd)
        liq = -p.liq_vol_load * h + w
        liq_c = liq - 0.5 * (p.liq_vol_load**2 * p.h_sd**2 + p.liq_sd**2)
        eta = eta_nom * p.eta_mult * np.exp(-liq_c)
        depth = p.depth_obs_mult * np.exp(liq + rng.normal(0, p.depth_obs_noise, L))
        m = _ar1(rng, L, p.volu_phi, p.volu_sd)
        logv = p.volu_vol_load * h + m + p.volu_abs_ret_load * (np.abs(eps) - eabs)
        logv -= 0.5 * (p.volu_vol_load**2 * p.h_sd**2 + p.volu_sd**2)
        vol = p.vol_bin * p.volume_mult * np.exp(logv)
        bridge[b] = rng.normal(0, 1, N_STEPS)
        for k, arr in (("sigma", sigma), ("alpha", alpha), ("signal", signal), ("ret", ret),
                       ("eps", eps), ("half_spread", hs), ("eta", eta), ("depth_obs", depth),
                       ("volume", vol)):
            out[k][b] = arr
    return Market(
        p=p,
        sigma_nom=nom[:, 0],
        hs_nom=nom[:, 1],
        eta_nom=nom[:, 2],
        participation=nom[:, 3] * p.participation_mult,
        sigma_bar=nom[:, 0] * p.sigma_mult,
        eta_bar=nom[:, 2] * p.eta_mult,
        vbar=np.full(B, p.vol_bin * p.volume_mult),
        bridge_z=bridge,
        **out,
    )


def _lgamma(x: float) -> float:
    from math import lgamma

    return lgamma(x)


# --------------------------------------------------------------------------------------
# Observation features (all causal; normalised by the trader's *nominal* references)
# --------------------------------------------------------------------------------------
EWMA_HALFLIFE = 10.0  # bins


def market_features(m: Market) -> dict[str, np.ndarray]:
    """Causal market features at every index k, shape (B, L).  Uses only data up to k."""
    lam = 1 - 0.5 ** (1 / EWMA_HALFLIFE)
    B, L = m.ret.shape
    rv2 = np.empty((B, L))
    lv = np.empty((B, L))
    rv2[:, 0] = m.sigma_nom**2
    lv[:, 0] = np.log(m.p.vol_bin)
    for k in range(1, L):
        rv2[:, k] = (1 - lam) * rv2[:, k - 1] + lam * m.ret[:, k] ** 2
        lv[:, k] = (1 - lam) * lv[:, k - 1] + lam * np.log(m.volume[:, k])
    sn = m.sigma_nom[:, None]
    return {
        "signal": m.signal,
        "log_rv": 0.5 * np.log(rv2 / sn**2),
        "log_spread": np.log(m.half_spread / m.hs_nom[:, None]),
        "log_depth": np.log(m.depth_obs),
        "log_volume": lv - np.log(m.p.vol_bin),
        "last_ret": m.ret / sn,
    }


FEATURE_NAMES = ("inv", "time", "signal", "log_rv", "log_spread", "log_depth", "log_volume",
                 "last_ret", "size")
OBS_DIM = len(FEATURE_NAMES)


def build_obs(feats: dict[str, np.ndarray], k: int, inv: np.ndarray, t: int,
              participation: np.ndarray, ref_participation: float) -> np.ndarray:
    B = inv.shape[0]
    obs = np.empty((B, OBS_DIM), dtype=np.float32)
    obs[:, 0] = inv
    obs[:, 1] = (N_STEPS - t) / N_STEPS
    obs[:, 2] = feats["signal"][:, k]
    obs[:, 3] = feats["log_rv"][:, k]
    obs[:, 4] = feats["log_spread"][:, k]
    obs[:, 5] = feats["log_depth"][:, k]
    obs[:, 6] = feats["log_volume"][:, k]
    obs[:, 7] = np.clip(feats["last_ret"][:, k], -10, 10)
    obs[:, 8] = np.log(participation / ref_participation)
    return obs


# --------------------------------------------------------------------------------------
# Execution engine
# --------------------------------------------------------------------------------------
REF_PARTICIPATION = 0.075  # observation normaliser for order size (nominal midpoint)


class Execution:
    """Step-wise execution of B parent sell orders of normalised size 1 (fraction of X)."""

    def __init__(self, m: Market):
        self.m = m
        self.feats = market_features(m)
        self.p = m.p
        B = m.B
        self.X = m.participation * N_STEPS * m.vbar / m.p.volume_mult  # shares (nominal volume)
        self.q = np.ones(B)  # remaining fraction of X
        self.t = 0
        self.cash = np.zeros(B)  # proceeds per unit of X * M_0
        self.perm = np.zeros(B)
        self.cum = np.zeros(B)  # cumulative shares executed / realised mean bin volume
        self.mart = np.zeros(B)  # zero-mean price-innovation part of the P&L (control variate)
        self.trans = np.zeros(B)
        self.F = np.ones(B)  # unaffected fundamental at current decision time
        self.trades = np.zeros((B, N_STEPS))
        self.f_lat = min(self.p.latency_s / DT_SECONDS, 1.0)
        p = self.p
        if p.impact_model == "linear":
            self.temp_scale = 1.0 / np.sqrt(p.impact_ref_participation)
        elif p.impact_model == "transient":
            self.temp_scale = 1.0 - p.transient_share
        else:
            self.temp_scale = 1.0
        self.decay = np.exp(-DT_SECONDS / p.transient_tau_s)
        # transient injections are scaled so that, at a constant TWAP rate, the stationary
        # transient deviation paid equals the removed instantaneous share (matched cost)
        self.trans_gain = p.transient_share * (1 - self.decay)

    @property
    def k(self) -> int:
        return WINDOW + self.t

    @property
    def done(self) -> bool:
        return self.t >= N_STEPS

    def mid(self) -> np.ndarray:
        return self.F * (1.0 - self.perm - self.trans)

    def obs(self) -> np.ndarray:
        return build_obs(self.feats, self.k, self.q, self.t, self.m.participation,
                         REF_PARTICIPATION)

    def step(self, v: np.ndarray, denoise: bool = False) -> np.ndarray:
        """Sell fraction `v` of X (clipped to inventory; final step liquidates).

        Returns the per-step reward in bp of arrival notional; rewards sum to -IS.
        With `denoise=True` the exogenous return term is replaced by its conditional
        mean given the full simulator state (unbiased control variate, training only).
        """
        m, p, k, t = self.m, self.p, self.k, self.t
        v = np.clip(v, 0.0, self.q)
        if t == N_STEPS - 1:
            v = self.q.copy()
        mid0 = self.mid()
        sig = m.sigma[:, k]
        beta = p.alpha_beta * p.alpha_beta_mult
        # unaffected price at fill time t + latency (Brownian bridge inside the bin)
        f = self.f_lat
        eps_next = m.eps[:, k + 1]
        part_noise = f * eps_next + np.sqrt(max(f * (1 - f), 0.0)) * m.bridge_z[:, t]
        F_fill = self.F * (1 + sig * (beta * m.alpha[:, k] * f + np.sqrt(1 - beta**2) * part_noise))
        shares = v * self.X
        bin_part = shares / m.volume[:, k]
        if p.impact_model == "linear":
            temp = m.eta[:, k] * sig * self.temp_scale * bin_part
        else:
            temp = m.eta[:, k] * sig * self.temp_scale * np.sqrt(bin_part)
        mid_fill = F_fill * (1 - self.perm - self.trans)
        price = mid_fill * (1 - m.half_spread[:, k] - temp)
        proceeds = v * price
        self.cash += proceeds
        inno = self.F * sig * np.sqrt(1 - beta**2)  # scale of the exogenous innovation
        self.mart += v * inno * part_noise * (1 - self.perm - self.trans) * (
            1 - m.half_spread[:, k] - temp)
        # impact state updates
        # cumulative impact: concave (square-root law, Toth et al. 2011; Donier et al. 2015)
        # for "sqrt"/"transient"; linear (Almgren-Chriss) for "linear", matched at the
        # reference metaorder size.  Sell-only actions rule out round-trip manipulation.
        self.cum = self.cum + shares / m.vbar
        if p.impact_model == "linear":
            q_ref = p.impact_ref_participation * N_STEPS
            self.perm = p.perm_coef * m.sigma_bar * self.cum / np.sqrt(q_ref)
        else:
            self.perm = p.perm_coef * m.sigma_bar * np.sqrt(self.cum)
        if p.impact_model == "transient":
            inj = m.eta[:, k] * sig * np.sqrt(bin_part) * self.trans_gain
            self.trans = self.trans * self.decay + inj
        q_next = self.q - v
        self.mart += q_next * inno * eps_next * (1.0 - self.perm - self.trans)
        F_next = self.F * (1 + m.ret[:, k + 1])
        if denoise:
            F_next_r = self.F * (1 + sig * beta * m.alpha[:, k])
        else:
            F_next_r = F_next
        mid1 = F_next_r * (1.0 - self.perm - self.trans)
        reward = 1e4 * (proceeds - v * mid0 + q_next * (mid1 - mid0))
        self.trades[:, t] = v
        self.F = F_next
        self.q = q_next
        self.t += 1
        return reward

    def shortfall_bps(self) -> np.ndarray:
        assert self.done
        return 1e4 * (1.0 - self.cash)

    def shortfall_cv_bps(self) -> np.ndarray:
        """Control-variate IS: realised IS minus its zero-mean price-innovation component.

        Each subtracted term is (quantity fixed at decision time t) x (innovation drawn after
        t), so it has conditional mean zero and E[IS_cv] = E[IS] exactly (tower property).
        Uses latent simulator quantities, so it is used only for *labels* and variance
        reduction, never as an input to any policy or gate.
        """
        assert self.done
        return 1e4 * (1.0 - self.cash + self.mart)


def rollout(m: Market, policy, denoise: bool = False) -> dict:
    """Roll out `policy(ex) -> v` (fractions of X). Returns IS, control-variate IS, trades."""
    ex = Execution(m)
    if hasattr(policy, "reset"):
        policy.reset(ex)
    while not ex.done:
        ex.step(policy(ex), denoise=denoise)
    return {"IS": ex.shortfall_bps(), "IS_cv": ex.shortfall_cv_bps(), "trades": ex.trades}


def run_policy(m: Market, policy, denoise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Roll out `policy` on every episode. Returns (IS_bps, trades)."""
    r = rollout(m, policy, denoise)
    return r["IS"], r["trades"]
