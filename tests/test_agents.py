"""Policy interface: adapters are bit-identical to the frozen study policies, the factory
builds every kind, and the rolled-out gated policy equals the study's analytical shortcut."""

import numpy as np
import pytest
from conftest import simval_market

from adexec import interfaces as I
from adexec import policies as frozen
from adexec.agents import (
    ACPolicy,
    ADAPTPolicy,
    BlendPolicy,
    GatedPolicy,
    PolicyContext,
    TWAPPolicy,
    describe_policy,
    make_policy,
)
from adexec.evaluation.engine import rollout
from adexec.sim import rollout as frozen_rollout

W = [1.5, 0.5, 1.25, -0.75, 0.25]


@pytest.mark.parametrize("new,old", [
    (TWAPPolicy(), frozen.TWAP()),
    (ACPolicy(1.3), frozen.AlmgrenChriss(1.3)),
    (ACPolicy(-2.0), frozen.AlmgrenChriss(-2.0)),
    (ADAPTPolicy(W), frozen.Adaptive(*W)),
    (BlendPolicy(ADAPTPolicy(W), TWAPPolicy(), 0.25),
     frozen.Blend(frozen.Adaptive(*W), frozen.TWAP(), 0.25)),
])
def test_adapters_are_bit_identical_to_frozen_policies(new, old):
    m = simval_market(150)
    a, b = rollout(m, new), frozen_rollout(m, old)
    assert np.array_equal(a.is_bps, b["IS"]) and np.array_equal(a.trades, b["trades"])
    assert np.array_equal(a.is_cv_bps, b["IS_cv"])


def test_classical_policies_satisfy_protocols():
    for p in (TWAPPolicy(), ACPolicy(0.5), ADAPTPolicy(W)):
        assert isinstance(p, I.ExecutionPolicy) and isinstance(p, I.BaselinePolicy)
        assert isinstance(describe_policy(p)["parameters"], dict)


def test_adapt_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError, match="5 weights"):
        ADAPTPolicy([1, 2, 3])


def test_factory_builds_every_classical_kind_and_tuned_values():
    ctx = PolicyContext()  # tuned values: results/baselines_sqrt.json (committed)
    ac = make_policy({"kind": "ac", "tuned": True}, ctx)
    ad = make_policy({"kind": "adapt", "tuned": True}, ctx)
    tuned = ctx.tuned_baselines()
    assert ac.kappa == tuned["AC"]["kappa"] and ad.weights == tuned["ADAPT"]["weights"]
    bl = make_policy({"kind": "blend", "w": 0.5, "first": {"kind": "twap"},
                      "second": {"kind": "ac", "kappa": 1.0}, "name": "B"})
    assert bl.name == "B"
    with pytest.raises(ValueError, match="unknown policy kind"):
        make_policy({"kind": "momentum"})


def test_missing_checkpoints_give_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        make_policy({"kind": "ppo", "replicate": 0}, PolicyContext(models_dir=tmp_path))


def test_ppo_policy_wraps_frozen_ensemble(tiny_ppo, tiny_models):
    from adexec.rl import PPOEnsemble

    m = simval_market(64)
    ref = PPOEnsemble([tiny_models / f"sqrt_r00_m{j}.zip" for j in range(2)])
    assert np.array_equal(rollout(m, tiny_ppo).is_bps, frozen_rollout(m, ref)["IS"])
    assert isinstance(tiny_ppo, I.RLPolicy)
    obs = np.zeros((5, 9), np.float32)
    acts = tiny_ppo.member_actions(obs)
    assert acts.shape == (2, 5) and (np.abs(acts) <= 1).all()


def test_gated_rollout_equals_analytical_where(tiny_ppo):
    """The study never rolls a gate out; it computes where(fallback, IS_FB, IS_RL). This is
    valid only if a once-per-episode decision makes the episodes independent; check it."""
    m = simval_market(300)
    rng = np.random.default_rng(3)
    mask = rng.random(m.B) < 0.4
    fb = TWAPPolicy()
    g = GatedPolicy(tiny_ppo, fb, lambda mk: mask)
    got = rollout(m, g)
    rl, tw = rollout(m, tiny_ppo), rollout(m, fb)
    assert np.allclose(got.is_bps, np.where(mask, tw.is_bps, rl.is_bps), atol=1e-10)
    assert np.allclose(got.is_cv_bps, np.where(mask, tw.is_cv_bps, rl.is_cv_bps), atol=1e-10)
    assert g.fallback_rate == pytest.approx(mask.mean())


def test_gated_policy_validates_decision_shape(tiny_ppo):
    m = simval_market(10)
    g = GatedPolicy(tiny_ppo, TWAPPolicy(), lambda mk: np.zeros(3, bool))
    with pytest.raises(ValueError, match="shape"):
        rollout(m, g)
    with pytest.raises(RuntimeError, match="before reset"):
        GatedPolicy(tiny_ppo, TWAPPolicy(), lambda mk: mk).act(None)
