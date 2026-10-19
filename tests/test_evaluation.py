"""Policy-agnostic evaluation engine."""

import ast
from pathlib import Path

import numpy as np
from conftest import simval_market

from adexec import policies as frozen
from adexec.agents import TWAPPolicy
from adexec.evaluation.engine import evaluate_policies, evaluate_scenario, rollout
from adexec.market import Scenario
from adexec.sim import rollout as frozen_rollout

PKG = Path(__file__).resolve().parent.parent / "adexec"


def test_engine_matches_frozen_rollout_for_plain_callables():
    m = simval_market(120)
    pol = frozen.Adaptive(1.0, 0.5, 0.5, 0, 0)  # a frozen class: no act(), only __call__
    a, b = rollout(m, pol), frozen_rollout(m, pol)
    assert np.array_equal(a.is_bps, b["IS"]) and np.array_equal(a.is_cv_bps, b["IS_cv"])


def test_every_policy_sees_the_same_market():
    m = simval_market(80)
    res = evaluate_policies(m, {"a": TWAPPolicy(), "b": TWAPPolicy()})
    assert np.array_equal(res["a"].is_bps, res["b"].is_bps)


def test_evaluate_scenario_is_deterministic_and_seed_dependent():
    s = Scenario("LIQ-mod")
    a = evaluate_scenario(s, {"T": TWAPPolicy()}, 50, seed=1)["T"].is_bps
    b = evaluate_scenario(s, {"T": TWAPPolicy()}, 50, seed=1)["T"].is_bps
    c = evaluate_scenario(s, {"T": TWAPPolicy()}, 50, seed=2)["T"].is_bps
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_orders_complete_and_never_buy():
    m = simval_market(100)
    for pol in (frozen.TWAP(), frozen.AlmgrenChriss(-3.0), frozen.Adaptive(3, 3, 3, 3, 3)):
        tr = rollout(m, pol).trades
        assert np.allclose(tr.sum(axis=1), 1.0) and (tr >= -1e-12).all()


def _string_constants(path: Path) -> set[str]:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_engine_and_runner_contain_no_policy_specific_logic():
    names = {"TWAP", "AC", "ADAPT", "RL", "RL1", "RCG", "FB", "twap", "ppo", "adapt", "gated"}
    for rel in ("evaluation/engine.py", "benchmarks/runner.py"):
        src = (PKG / rel).read_text()
        assert not names & _string_constants(PKG / rel), rel
        assert "isinstance(" not in src, rel
