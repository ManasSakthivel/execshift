"""Market layer: scenarios reproduce the study's conditions; impact/latency invariants."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import simval_market

from adexec import interfaces as I
from adexec.config import CONDITION_BY_NAME, CONDITIONS, IMPACT_MODELS, NOMINAL, SPLIT_TAGS
from adexec.market import (
    BENCHMARK_STREAM_TAG,
    DEFAULT_SIMULATOR,
    LatencyModel,
    Scenario,
    benchmark_rngs,
    preregistered_scenarios,
)
from adexec.market import (
    IMPACT_MODELS as SPECS,
)
from adexec.policies import TWAP
from adexec.sim import rollout


def _study_condition_params(cond, impact_model, latency_s=None):
    """The rule from the pre-refactor runner (adexec/experiment.py at commit 17e688d)."""
    p = replace(CONDITION_BY_NAME[cond].params, impact_model=impact_model)
    if latency_s is not None and CONDITION_BY_NAME[cond].family != "LATENCY":
        p = replace(p, latency_s=latency_s)
    return p


@pytest.mark.parametrize("im", IMPACT_MODELS)
@pytest.mark.parametrize("lat", [None, 0.0, 1.0, 5.0])
def test_scenarios_reproduce_study_condition_params(im, lat):
    for c in CONDITIONS:
        assert Scenario(c.name, im, lat).params() == _study_condition_params(c.name, im, lat)


def test_preregistered_suite_is_complete_and_unchanged():
    sc = preregistered_scenarios()
    assert [s.condition for s in sc] == [c.name for c in CONDITIONS]
    assert len(sc) == 13 and all(s.is_preregistered for s in sc)
    assert {s.family for s in sc} == {"ID", "VOL", "LIQ", "ALPHA", "IMPACT", "LATENCY", "SIZE"}


def test_overrides_create_non_preregistered_scenarios_and_are_validated():
    s = Scenario("ID", overrides={"sigma_mult": 2.0})
    assert not s.is_preregistered and s.params().sigma_mult == 2.0
    assert s.fingerprint() != Scenario("ID").fingerprint()
    with pytest.raises(ValueError, match="regime knobs"):
        Scenario("ID", overrides={"perm_coef": 0.0})
    with pytest.raises(ValueError, match="unknown condition"):
        Scenario("VOL-extreme")
    with pytest.raises(ValueError, match="impact model"):
        Scenario("ID", "cubic")


def test_latency_model():
    assert LatencyModel("x", 3.0).apply(NOMINAL).latency_s == 3.0
    with pytest.raises(ValueError):
        LatencyModel("x", -1.0).apply(NOMINAL)
    assert isinstance(LatencyModel("x", 1.0), I.LatencyModel)


def test_simulator_and_impact_specs_satisfy_protocols():
    assert isinstance(DEFAULT_SIMULATOR, I.MarketSimulator)
    for spec in SPECS.values():
        assert isinstance(spec, I.ImpactModel)
        assert spec.apply(NOMINAL).impact_model == spec.name


def test_benchmark_stream_is_disjoint_from_study_streams():
    assert BENCHMARK_STREAM_TAG not in SPLIT_TAGS.values()
    a = DEFAULT_SIMULATOR.generate(NOMINAL, benchmark_rngs(Scenario("ID"), 5, 0))
    b = DEFAULT_SIMULATOR.generate(NOMINAL, benchmark_rngs(Scenario("ID"), 5, 1))
    assert not np.array_equal(a.ret, b.ret)


def test_benchmark_episodes_are_common_across_impact_models():
    """Robustness comparisons across impact models use the same exogenous randomness."""
    r = [benchmark_rngs(Scenario("ID", im), 3, 0)[0].normal() for im in IMPACT_MODELS]
    assert r[0] == r[1] == r[2]


@pytest.mark.parametrize("im", IMPACT_MODELS)
def test_engine_follows_declared_temporary_impact_exponent(im):
    """Quadrupling participation multiplies temporary cost by 4**exponent (0.5 or 1)."""
    costs = []
    for mult in (1.0, 4.0):
        p = replace(NOMINAL, impact_model=im, perm_coef=0.0, half_spread_mult=1e-9,
                    participation_mult=mult, latency_s=0.0)
        costs.append(rollout(simval_market(1500, p, cond=50), TWAP())["IS_cv"].mean())
    slope = np.log(costs[1] / costs[0]) / np.log(4)
    assert slope == pytest.approx(SPECS[im].temporary_exponent, abs=0.08)


def test_cost_increases_with_impact_coefficient_and_spread():
    base = rollout(simval_market(800), TWAP())["IS_cv"].mean()
    for kw in ({"eta_mult": 2.0}, {"half_spread_mult": 2.0}):
        assert rollout(simval_market(800, replace(NOMINAL, **kw)), TWAP())["IS_cv"].mean() > base


def test_latency_adds_variance_not_mean_cost_for_twap():
    a = rollout(simval_market(3000), TWAP())
    b = rollout(simval_market(3000, replace(NOMINAL, latency_s=10.0)), TWAP())
    assert abs(a["IS_cv"].mean() - b["IS_cv"].mean()) < 0.5
    assert b["IS"].std() >= a["IS"].std() * 0.98
