"""Uncertainty estimators equal the frozen gate-feature columns they wrap."""

import numpy as np
import pytest
from conftest import simval_market

from adexec import interfaces as I
from adexec.agents import ADAPTPolicy, TWAPPolicy, make_policy
from adexec.config import NOMINAL
from adexec.features import FEATURE_COLS, OODModel, gate_features, market_summary
from adexec.uncertainty import (
    EnsembleDisagreement,
    OODDistance,
    RCGScore,
    ShadowRegret,
    make_estimator,
)

COL = {c: i for i, c in enumerate(FEATURE_COLS)}


@pytest.fixture(scope="module")
def ref_market():
    return simval_market(400, cond=7)


def test_ood_distance_equals_feature_column(ref_market, tiny_ppo):
    m = simval_market(120, cond=8)
    est = OODDistance(reference=ref_market)
    ood = OODModel.fit(market_summary(ref_market))
    F = gate_features(m, tiny_ppo.ensemble, TWAPPolicy(), ood, "sqrt")
    assert np.allclose(est.score(m), F[:, COL["ood_maha"]])
    assert isinstance(est, I.UncertaintyEstimator)


def test_ensemble_and_shadow_equal_feature_columns(ref_market, tiny_ppo):
    m = simval_market(80, cond=9)
    ood = OODModel.fit(market_summary(ref_market))
    F = gate_features(m, tiny_ppo.ensemble, TWAPPolicy(), ood, "sqrt")
    assert np.allclose(EnsembleDisagreement(tiny_ppo).score(m), F[:, COL["u_ens_s0"]])
    assert np.allclose(ShadowRegret(tiny_ppo, TWAPPolicy()).score(m),
                       F[:, COL["shadow_regret"]])


def test_shadow_regret_of_a_policy_against_itself_is_zero():
    m = simval_market(50)
    assert np.allclose(ShadowRegret(TWAPPolicy(), TWAPPolicy()).score(m), 0.0)
    assert not np.allclose(ShadowRegret(ADAPTPolicy([1, 1, 1, 0, 0]), TWAPPolicy()).score(m), 0)


def test_rcg_score_uses_a_fitted_regressor(ref_market, tiny_ppo):
    class Const:
        def predict(self, F):
            return np.full(F.shape[0], 0.7)

    ood = OODModel.fit(market_summary(ref_market))
    s = RCGScore(Const(), tiny_ppo, TWAPPolicy(), ood).score(simval_market(20))
    assert np.allclose(s, 0.7)


def test_make_estimator_errors_and_gated_spec_threshold():
    with pytest.raises(ValueError, match="RCG needs"):
        make_estimator("rcg")
    g = make_policy({"kind": "gated", "estimator": "shadow", "threshold": 0.0,
                     "primary": {"kind": "adapt", "weights": [1, 0, 0, 0, 0]},
                     "fallback": {"kind": "twap"}})
    from adexec.evaluation.engine import rollout

    rollout(simval_market(60), g)
    assert 0.0 <= g.fallback_rate <= 1.0


def test_train_quantile_reference_is_the_train_split():
    est = OODDistance(impact_model="sqrt")  # 4,000 TRAIN-ID episodes, as in the study
    q = est.train_quantile(0.95)
    assert np.mean(est.train_scores() > q) == pytest.approx(0.05, abs=0.002)
    assert est.impact_model == NOMINAL.impact_model
