"""Metrics engine: hand-computed values, agreement with the frozen study code, units."""

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from adexec import analysis
from adexec import stats as frozen_stats
from adexec.evaluation import metrics as M


def test_implementation_shortfall_of_a_sell_order():
    q = np.array([[0.5, 0.5]])
    # sold half at 0.999 and half at 0.997 vs arrival mid 1 -> 20 bp
    assert M.implementation_shortfall_bps(q, np.array([[0.999, 0.997]]))[0] == pytest.approx(20)
    assert M.implementation_shortfall_bps(q, np.array([[1.0, 1.0]]))[0] == pytest.approx(0)
    with pytest.raises(ValueError):
        M.implementation_shortfall_bps(q, np.ones((1, 3)))


def test_location_and_dispersion():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    assert M.mean_cost(x) == 4.0 and M.median_cost(x) == 2.5
    assert M.std_cost(x) == pytest.approx(np.std(x))
    assert M.quantile_cost(x, 0.5) == 2.5
    with pytest.raises(ValueError):
        M.quantile_cost(x, 1.5)


def test_cvar_by_hand_and_against_frozen_describe():
    x = np.arange(1, 101, dtype=float)  # quantile(0.95) = 95.05 -> tail {96..100}
    assert M.cvar(x, 0.95) == pytest.approx(98.0)
    rng = np.random.default_rng(0)
    X = rng.standard_t(4, (3, 2000))
    want = analysis.describe(X)["cvar95"]
    assert np.mean([M.cvar(row, 0.95) for row in X]) == pytest.approx(want, rel=1e-12)
    with pytest.raises(ValueError):
        M.cvar(x, 1.0)


def test_cvar_is_at_least_var_and_mean():
    x = np.random.default_rng(1).lognormal(size=5000)
    assert M.cvar(x, 0.95) >= M.quantile_cost(x, 0.95) >= M.mean_cost(x)


def test_paired_metrics():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 1.0, 5.0])
    assert np.array_equal(M.paired_difference(a, b), [-1, 0, 2, -1])
    assert M.win_rate(a, b) == pytest.approx((2 + 0.5) / 4)  # tie counts one half
    assert np.array_equal(M.regret(a, b), a - b)
    with pytest.raises(ValueError, match="equal shape"):
        M.paired_difference(a, b[:3])


def test_relative_degradation_and_did():
    assert M.relative_degradation(12.0, 10.0) == pytest.approx(0.2)
    assert np.isnan(M.relative_degradation(1.0, 0.0))
    # edge -2 in reference, -0.5 under shift -> eroded by 1.5 bp
    assert M.difference_in_differences(np.full(4, 9.5), np.full(4, 10.0),
                                       np.full(4, 8.0), np.full(4, 10.0)) == pytest.approx(1.5)


def test_mean_ci_and_bootstrap_ci():
    x = np.random.default_rng(2).normal(1.0, 1.0, 400)
    m, lo, hi = M.mean_ci(x)
    assert lo < m < hi and hi - lo == pytest.approx(2 * 1.966 * x.std(ddof=1) / 20, rel=0.01)
    assert M.bootstrap_ci(x, seed=5) == M.bootstrap_ci(x, seed=5)  # seeded
    p, blo, bhi = M.bootstrap_ci(x, B=4000, seed=1)
    assert abs(blo - lo) < 0.03 and abs(bhi - hi) < 0.03
    assert np.isnan(M.mean_ci(np.array([1.0]))[1])


def test_association_metrics_agree_with_references():
    rng = np.random.default_rng(3)
    s = rng.normal(size=1000)
    r = 0.3 * s + rng.normal(size=1000)
    assert M.spearman(s, r) == pytest.approx(sps.spearmanr(s, r)[0])
    assert M.pearson(s, r) == pytest.approx(sps.pearsonr(s, r)[0])
    lab = r > 0
    assert M.auroc(s, lab) == pytest.approx(roc_auc_score(lab, s))
    assert M.auroc(s, lab) == frozen_stats.auroc(s, lab)
    assert M.auprc(s, lab) == frozen_stats.auprc(s, lab)
    assert M.auroc(-s, lab) == pytest.approx(1 - M.auroc(s, lab))


def test_risk_coverage_endpoints():
    s = np.arange(10.0)
    reg = np.array([-1.0] * 5 + [2.0] * 5)
    cov, sel, tot = M.risk_coverage(s, reg, 11)
    assert tot[0] == 0.0 and tot[-1] == pytest.approx(reg.mean())
    assert tot[5] == pytest.approx(-0.5)  # trusting only the five safe episodes


def test_metric_registry_is_documented():
    for name, m in M.METRICS.items():
        assert m.name == name and m.unit == "bp" and m.description
        assert m.higher_is_better is False
        assert isinstance(m(np.array([1.0, 2.0, 3.0])), float)
