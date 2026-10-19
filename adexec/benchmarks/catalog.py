"""The standard benchmark suites.

Every suite draws episodes from the post-study benchmark stream (split tag 66), so no suite
reuses a study episode or can reach the confirmatory test stream (tag 33). Results are
therefore NEW / POST-STUDY / EXPLORATORY by construction and are labelled so.

The 12 pre-registered shift conditions appear unchanged (definitions from the frozen
``adexec/config.py``). Scenarios outside the pre-registered suite are named ``... (NEW)``
and have ``is_preregistered == False``.

| suite        | category             | needs PPO checkpoints |
|--------------|----------------------|-----------------------|
| smoke        | smoke                | no (CI)               |
| baseline     | baseline             | yes                   |
| classical    | baseline             | no                    |
| seeds        | in-distribution      | yes (10 replicates)   |
| shift        | distribution-shift   | yes                   |
| impact       | impact-robustness    | yes (all 3 models)    |
| latency      | latency-robustness   | yes                   |
| liquidity    | liquidity-robustness | yes                   |
| risk         | risk                 | yes                   |
| gate         | gate                 | yes                   |
| robustness   | composite            | yes (all 3 models)    |
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import CONDITIONS, IMPACT_MODELS, N_REPLICATES, SHIFTS
from ..market import Scenario
from .suite import BenchmarkSuite, named_policies

CLASSICAL = [{"kind": "twap"}, {"kind": "ac", "tuned": True}, {"kind": "adapt", "tuned": True}]
RL = [{"kind": "ppo", "replicate": 0}]
ALL_CONDITIONS = tuple(c.name for c in CONDITIONS)
LATENCY_GRID_S = (0.0, 0.2, 1.0, 2.0, 5.0, 10.0)  # 0.2 s is nominal; 2 and 10 s are LAT-mod/sev

# A NEW, harsher liquidity point beyond the pre-registered LIQ-sev (not part of the study).
LIQ_EXTREME = {"half_spread_mult": 6.0, "eta_mult": 3.5, "depth_obs_mult": 1 / 3.5,
               "volume_mult": 0.35}


def _scen(conds, impact="sqrt"):
    return tuple(Scenario(c, impact) for c in conds)


def smoke() -> BenchmarkSuite:
    return BenchmarkSuite(
        "smoke", "smoke", "Classical policies on ID and LIQ-sev at toy scale (CI, no PPO).",
        named_policies(CLASSICAL), _scen(("ID", "LIQ-sev")), n_episodes=200)


def classical() -> BenchmarkSuite:
    return BenchmarkSuite(
        "classical", "baseline",
        "TWAP vs tuned Almgren-Chriss vs tuned ADAPT on all 13 conditions (no PPO needed).",
        named_policies(CLASSICAL), _scen(ALL_CONDITIONS))


def baseline() -> BenchmarkSuite:
    return BenchmarkSuite(
        "baseline", "baseline",
        "RL (5-member ensemble and single member) vs tuned classical baselines, "
        "in-distribution.",
        named_policies([*CLASSICAL, *RL, {"kind": "ppo", "replicate": 0,
                                          "single_member": True}]),
        _scen(("ID",)), n_episodes=4000)


def seeds() -> BenchmarkSuite:
    reps = [{"kind": "ppo", "replicate": r} for r in range(N_REPLICATES)]
    return BenchmarkSuite(
        "seeds", "in-distribution",
        "All 10 RL training replicates vs TWAP in-distribution: training-seed variability.",
        named_policies([{"kind": "twap"}, *reps]), _scen(("ID",)), n_episodes=2000)


def shift() -> BenchmarkSuite:
    return BenchmarkSuite(
        "shift", "distribution-shift",
        "RL vs classical baselines on ID and the 12 pre-registered shift conditions.",
        named_policies([*CLASSICAL, *RL]), _scen(ALL_CONDITIONS))


def impact() -> BenchmarkSuite:
    conds = ("ID", "LIQ-sev", "IMPACT-mod", "IMPACT-sev", "SIZE-sev")
    return BenchmarkSuite(
        "impact", "impact-robustness",
        "The same conditions under the sqrt, linear and transient impact models. Policies "
        "are rebuilt per model: tuned baselines and PPO checkpoints are impact-specific.",
        named_policies([*CLASSICAL, *RL]),
        tuple(Scenario(c, im) for im in IMPACT_MODELS for c in conds))


def latency() -> BenchmarkSuite:
    scen = tuple(Scenario("ID", "sqrt", lat, label=f"ID/sqrt/lat{lat:g}s")
                 for lat in LATENCY_GRID_S)
    return BenchmarkSuite(
        "latency", "latency-robustness",
        "In-distribution market with order-to-fill latency from 0 to 10 s.",
        named_policies([*CLASSICAL, *RL]), scen,
        notes=("Latency overrides on ID are not pre-registered scenarios. The study's "
               "LAT-mod / LAT-sev conditions are 2 s and 10 s.",))


def liquidity() -> BenchmarkSuite:
    scen = (*_scen(("ID", "LIQ-mod", "LIQ-sev")),
            Scenario("ID", "sqrt", overrides=LIQ_EXTREME, label="LIQ-extreme (NEW)"))
    return BenchmarkSuite(
        "liquidity", "liquidity-robustness",
        "Liquidity drought of increasing severity, including a NEW point beyond LIQ-sev.",
        named_policies([*CLASSICAL, *RL]), scen,
        notes=("LIQ-extreme is a new, post-study scenario: spreads x6, impact x3.5, visible "
               "depth / 3.5, volume x0.35.",))


def risk() -> BenchmarkSuite:
    return BenchmarkSuite(
        "risk", "risk",
        "Tail risk of RL vs TWAP and ADAPT across all 13 conditions (CVaR, VaR, dispersion).",
        named_policies([{"kind": "twap"}, {"kind": "adapt", "tuned": True}, *RL]),
        _scen(ALL_CONDITIONS), n_episodes=2000,
        metrics=("mean", "std", "p95", "cvar95", "cvar99"))


def gate() -> BenchmarkSuite:
    gated = [{"kind": "gated", "estimator": e, "quantile": 0.95, "primary": RL[0],
              "fallback": {"kind": "twap"}} for e in ("ood", "ensemble")]
    gated.append({"kind": "gated", "estimator": "shadow", "threshold": 0.0, "primary": RL[0],
                  "fallback": {"kind": "twap"}})
    return BenchmarkSuite(
        "gate", "gate",
        "Uncertainty-gated policies rolled out step by step: cost and intervention "
        "frequency per condition. Thresholds come from TRAIN-ID quantiles or 0 (shadow); "
        "nothing is tuned on these episodes.",
        named_policies([{"kind": "twap"}, *RL, *gated]), _scen(ALL_CONDITIONS),
        notes=("The study's learned RCG gate needs validation data and is evaluated only "
               "in the confirmatory pipeline; these gates are its single-signal "
               "components.",))


def robustness() -> BenchmarkSuite:
    return BenchmarkSuite(
        "robustness", "composite",
        "All 13 conditions x 3 impact models (39 scenarios): RL vs classical baselines.",
        named_policies([*CLASSICAL, *RL]),
        tuple(Scenario(c.name, im) for im in IMPACT_MODELS for c in (CONDITIONS[0], *SHIFTS)),
        n_episodes=500)


SUITES: dict[str, Callable[[], BenchmarkSuite]] = {
    "smoke": smoke, "classical": classical, "baseline": baseline, "seeds": seeds,
    "shift": shift, "impact": impact, "latency": latency, "liquidity": liquidity,
    "risk": risk, "gate": gate, "robustness": robustness,
}


def get_suite(name: str) -> BenchmarkSuite:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; available: {sorted(SUITES)}")
    return SUITES[name]()


__all__ = ["LATENCY_GRID_S", "LIQ_EXTREME", "SUITES", "get_suite"]
