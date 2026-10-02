"""Frozen experimental configuration.

Everything that defines *what is being measured* lives here: the simulator's nominal
(in-distribution) parameters, the pre-registered distribution-shift suite, the data
partitions and the primary endpoint.  Changing anything in this file after the
pre-registration lock (``research/prereg.lock``) invalidates the confirmatory run; the
integrity tests check the hash.

All market data in this project are SYNTHETIC.  Parameter values are chosen from ranges
reported in the empirical microstructure literature (see docs/SIMULATOR_VALIDATION.md for
the source of every number); they are not fitted to any particular asset.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------------------------
# Time grid
# --------------------------------------------------------------------------------------
DT_SECONDS = 30.0  # one decision interval
N_STEPS = 60  # execution horizon: 60 x 30 s = 30 minutes
WINDOW = 60  # pre-trade observation window (30 minutes) used by gates
BINS_PER_DAY = 780  # 6.5 h trading day / 30 s
SYNTHETIC = True  # every result in this repository is from simulated markets


@dataclass(frozen=True)
class MarketParams:
    """Parameters of one synthetic market regime.

    ``*_mult`` fields are the distribution-shift knobs: they scale the *realised* market
    while the trader's *nominal* reference values (what a desk would know from history)
    stay at their in-distribution values.
    """

    # cross-sectional heterogeneity of nominal stock characteristics (log-normal medians, log-sd)
    sigma_med: float = 5.5e-4  # per-30 s return vol (~24% annualised)
    sigma_logsd: float = 0.30
    half_spread_med: float = 1.5e-4  # 1.5 bp half-spread (3 bp quoted spread)
    half_spread_logsd: float = 0.40
    eta_med: float = 0.70  # temporary-impact coefficient (square-root law prefactor)
    eta_logsd: float = 0.25
    perm_coef: float = 0.50  # cumulative-impact prefactor: sigma * sqrt(executed / bin volume)
    vol_bin: float = 1000.0  # nominal shares traded per 30 s bin (normalisation)
    participation_lo: float = 0.03  # parent-order size as fraction of horizon volume
    participation_hi: float = 0.12

    # stochastic volatility (log-AR(1)) with Student-t return innovations
    h_phi: float = 0.97
    h_sd: float = 0.35
    t_dof: float = 4.0

    # latent short-term alpha (OU / AR(1)), observed through a noisy signal
    alpha_tau_s: float = 180.0  # alpha decorrelation time (s)
    alpha_beta: float = 0.15  # per-step predictive loading -> R^2(true alpha) ~ 2.3%
    signal_rho: float = 0.80  # corr(observed signal, true alpha)

    # spread / liquidity / volume state processes
    spread_vol_load: float = 0.50  # log half-spread loading on log-vol state
    spread_phi: float = 0.90
    spread_sd: float = 0.20
    liq_vol_load: float = 0.60  # log-depth loading on (minus) log-vol state
    liq_phi: float = 0.95
    liq_sd: float = 0.30
    depth_obs_noise: float = 0.15
    volu_vol_load: float = 0.80
    volu_phi: float = 0.80
    volu_sd: float = 0.30
    volu_abs_ret_load: float = 0.25

    # impact model: "sqrt" (base), "linear", "transient"
    impact_model: str = "sqrt"
    impact_ref_participation: float = 0.075  # where alternative models are matched to base
    transient_tau_s: float = 120.0
    transient_share: float = 0.5  # share of temporary impact that is transient (propagator)

    latency_s: float = 0.2  # order-to-fill latency

    # distribution-shift multipliers (1.0 = in-distribution)
    sigma_mult: float = 1.0
    half_spread_mult: float = 1.0
    eta_mult: float = 1.0
    depth_obs_mult: float = 1.0  # what the *observable* depth shows (liquidity drought is visible)
    volume_mult: float = 1.0
    alpha_beta_mult: float = 1.0
    participation_mult: float = 1.0


NOMINAL = MarketParams()


@dataclass(frozen=True)
class Condition:
    name: str
    family: str  # "ID" or shift family name
    severity: str  # "none" | "moderate" | "severe"
    params: MarketParams = field(default=NOMINAL)


def _shift(name: str, family: str, severity: str, **kw) -> Condition:
    return Condition(name, family, severity, replace(NOMINAL, **kw))


# --------------------------------------------------------------------------------------
# Pre-registered distribution-shift suite (frozen before any RL policy was trained).
# Magnitudes are anchored to documented market episodes; see PREREGISTRATION.md section 4.
# --------------------------------------------------------------------------------------
ID = Condition("ID", "ID", "none", NOMINAL)
SHIFTS: tuple[Condition, ...] = (
    # volatility regime: calm (~VIX 15) -> stressed (~VIX 26 / ~VIX 45)
    _shift("VOL-mod", "VOL", "moderate", sigma_mult=1.75),
    _shift("VOL-sev", "VOL", "severe", sigma_mult=3.0),
    # liquidity drought: spreads widen, books thin (visible), volume falls (cf. March 2020)
    _shift("LIQ-mod", "LIQ", "moderate", half_spread_mult=2.0, eta_mult=1.5,
           depth_obs_mult=1 / 1.5, volume_mult=0.7),
    _shift("LIQ-sev", "LIQ", "severe", half_spread_mult=4.0, eta_mult=2.5,
           depth_obs_mult=1 / 2.5, volume_mult=0.5),
    # alpha decay: the order-flow signal loses predictive power (crowding / post-publication decay)
    _shift("ALPHA-mod", "ALPHA", "moderate", alpha_beta_mult=0.5),
    _shift("ALPHA-sev", "ALPHA", "severe", alpha_beta_mult=0.0),
    # hidden impact shift: impact coefficient rises with no change in visible depth
    _shift("IMPACT-mod", "IMPACT", "moderate", eta_mult=2.0),
    _shift("IMPACT-sev", "IMPACT", "severe", eta_mult=3.0),
    # latency: cloud/retail API (2 s) and degraded infrastructure (10 s) vs 0.2 s nominal
    _shift("LAT-mod", "LATENCY", "moderate", latency_s=2.0),
    _shift("LAT-sev", "LATENCY", "severe", latency_s=10.0),
    # order size: parent orders 2x / 3x larger relative to volume
    _shift("SIZE-mod", "SIZE", "moderate", participation_mult=2.0),
    _shift("SIZE-sev", "SIZE", "severe", participation_mult=3.0),
)
CONDITIONS: tuple[Condition, ...] = (ID, *SHIFTS)
CONDITION_BY_NAME = {c.name: c for c in CONDITIONS}
FAMILIES: tuple[str, ...] = ("VOL", "LIQ", "ALPHA", "IMPACT", "LATENCY", "SIZE")

# Deployment mixture used by the primary gate endpoint: half in-distribution, half spread
# evenly across the 12 shift conditions.
MIXTURE_WEIGHTS = {"ID": 0.5, **{c.name: 0.5 / len(SHIFTS) for c in SHIFTS}}

# --------------------------------------------------------------------------------------
# Data partitions.  Every episode is generated from SeedSequence([SPLIT_TAG, cond, idx]); the
# split tag makes train / validation / test streams disjoint by construction.
# --------------------------------------------------------------------------------------
SPLIT_TAGS = {"train": 11, "val": 22, "test": 33, "rltrain": 44, "simval": 55}
N_EPISODES = {
    "train": {"ID": 4000},  # baseline tuning (ID only, like RL training)
    "val": {"ID": 2000, **{c.name: 500 for c in SHIFTS}},  # gate fitting / model selection
    "test": {"ID": 4000, **{c.name: 1000 for c in SHIFTS}},  # confirmatory, touched once
}

N_REPLICATES = 10  # independent RL training replicates (each an ensemble)
ENSEMBLE_SIZE = 5  # PPO members per replicate
IMPACT_MODELS = ("sqrt", "linear", "transient")
PRIMARY_IMPACT = "sqrt"
ALPHA = 0.05
