"""Unit tests for the simulator, policies, statistics and contamination guard."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from adexec import guard
from adexec.config import N_STEPS, NOMINAL, SPLIT_TAGS
from adexec.policies import TWAP, Adaptive, AlmgrenChriss, Blend
from adexec.sim import Execution, episode_rngs, generate_market, rollout, run_policy
from adexec.stats import auroc, crossed_bootstrap, holm, risk_coverage


def market(n=400, p=NOMINAL, cond=0):
    return generate_market(p, episode_rngs(SPLIT_TAGS["simval"], 900 + cond, np.arange(n)))


def test_generation_is_deterministic_per_seed():
    a, b = market(20), market(20)
    assert np.array_equal(a.ret, b.ret) and np.array_equal(a.eta, b.eta)
    c = market(20, cond=1)
    assert not np.array_equal(a.ret, c.ret)


def test_frictionless_flat_market_has_zero_shortfall():
    p = replace(NOMINAL, eta_mult=0.0, perm_coef=0.0, half_spread_mult=1e-12, sigma_mult=1e-9,
                latency_s=0.0)
    IS, trades = run_policy(market(50, p), TWAP())
    assert np.allclose(IS, 0.0, atol=1e-6)
    assert np.allclose(trades.sum(axis=1), 1.0)


def test_order_always_completes_and_never_buys():
    m = market(200)
    for pol in (TWAP(), AlmgrenChriss(2.0), AlmgrenChriss(-2.0), Adaptive(3, 1, 1, -1, 1)):
        _, tr = run_policy(m, pol)
        assert np.allclose(tr.sum(axis=1), 1.0) and (tr >= -1e-12).all()


def test_rewards_sum_to_minus_shortfall():
    m = market(100)
    ex = Execution(m)
    total = np.zeros(m.B)
    pol = Adaptive(1.0, 0.5, 0.5, 0, 0)
    while not ex.done:
        total += ex.step(pol(ex))
    assert np.allclose(total, -ex.shortfall_bps(), atol=1e-8)


def test_control_variate_is_unbiased_and_reduces_variance():
    m = market(4000)
    r = rollout(m, Adaptive(1.5, 0.5, 1.25, -0.75, 0.25))
    d = r["IS"] - r["IS_cv"]  # the subtracted martingale term: mean zero
    assert abs(d.mean()) < 3 * d.std() / np.sqrt(d.size)
    assert r["IS_cv"].std() < 0.6 * r["IS"].std()


def test_ac_limits():
    assert np.allclose(AlmgrenChriss(0.0).schedule, 1 / N_STEPS)
    front, back = AlmgrenChriss(2.0).schedule, AlmgrenChriss(-2.0).schedule
    assert front[0] > front[-1] and back[0] < back[-1]
    assert np.allclose(front, back[::-1])


def test_blend_is_convex_combination():
    m = market(50)
    _, t1 = run_policy(m, TWAP())
    _, t2 = run_policy(m, Blend(TWAP(), TWAP(), 0.3))
    assert np.allclose(t1, t2)


def test_impact_models_matched_at_reference_participation():
    costs = {}
    for im in ("sqrt", "linear", "transient"):
        p = replace(NOMINAL, impact_model=im, participation_lo=0.075, participation_hi=0.075)
        p0 = replace(p, eta_mult=0.0, perm_coef=0.0, half_spread_mult=1e-12)
        IS, _ = run_policy(market(2000, p), TWAP())
        IS0, _ = run_policy(market(2000, p0), TWAP())
        costs[im] = (IS - IS0).mean()
    assert costs["linear"] == pytest.approx(costs["sqrt"], rel=0.2)
    assert costs["transient"] == pytest.approx(costs["sqrt"], rel=0.1)


def test_holm_matches_definition():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    assert out["a"]["reject"] and not out["b"]["reject"]


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    y = rng.random(500) < 1 / (1 + np.exp(-s))
    assert auroc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_risk_coverage_endpoints():
    rng = np.random.default_rng(0)
    reg = rng.normal(size=100)
    cov, sel, tot = risk_coverage(rng.normal(size=100), reg, 11)
    assert tot[0] == 0 and tot[-1] == pytest.approx(reg.mean())


def test_bootstrap_signed_contrast():
    rng = np.random.default_rng(3)
    d = {"A": rng.normal(1.0, 1, (10, 2000)), "B": rng.normal(0.0, 1, (10, 2000))}
    r = crossed_bootstrap(d, {"A": 1.0, "B": -1.0}, B=1000, normalize=False)
    assert r["estimate"] == pytest.approx(1.0, abs=0.1) and r["ci_low"] > 0.8


def test_spearman_of_identical_scores_is_one():
    x = np.arange(10.0)
    assert sps.spearmanr(x, x)[0] == pytest.approx(1.0)


def test_lock_detects_modification(tmp_path, monkeypatch):
    f = tmp_path / "frozen.py"
    f.write_text("x = 1\n")
    pre = tmp_path / "PREREG.md"
    pre.write_text("plan\n")
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "PREREG", pre)
    monkeypatch.setattr(guard, "LOCK", tmp_path / "prereg.lock")
    monkeypatch.setattr(guard, "FROZEN", ("frozen.py",))
    guard.write_lock()
    assert json.loads((tmp_path / "prereg.lock").read_text())["files"]["frozen.py"]
    guard.check_lock()
    f.write_text("x = 2\n")
    with pytest.raises(guard.ContaminationError):
        guard.check_lock()
