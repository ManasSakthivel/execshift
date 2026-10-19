"""Benchmark framework: suite definitions, runner behaviour, recorded runs."""

import json

import numpy as np
import pytest

from adexec.benchmarks import (
    SUITES,
    BenchmarkSuite,
    get_suite,
    named_policies,
    record_run,
    run_suite,
)
from adexec.config import CONDITION_BY_NAME, CONDITIONS
from adexec.experiments.config import hash_config
from adexec.experiments.registry import FileArtifactStore
from adexec.market import Scenario


@pytest.mark.parametrize("name", sorted(SUITES))
def test_every_suite_is_well_formed(name):
    s = get_suite(name)
    assert s.name == name and s.reference in s.policies and s.scenarios
    d = s.to_dict()
    assert len(d["scenarios"]) == len(s.scenarios)
    assert all(len(sc["fingerprint"]) == 16 for sc in d["scenarios"])


def test_shift_suite_contains_the_12_frozen_shifts_unchanged():
    s = get_suite("shift")
    assert [x.condition for x in s.scenarios] == [c.name for c in CONDITIONS]
    for sc in s.scenarios:
        assert sc.is_preregistered
        assert sc.params() == CONDITION_BY_NAME[sc.condition].params


def test_new_scenarios_are_labelled_as_new():
    for name in SUITES:
        for sc in get_suite(name).scenarios:
            if sc.overrides:
                assert "NEW" in sc.name and not sc.is_preregistered


def test_robustness_suite_covers_all_conditions_under_three_impact_models():
    s = get_suite("robustness")
    assert len(s.scenarios) == 39
    assert {x.impact_model for x in s.scenarios} == {"sqrt", "linear", "transient"}


def test_suite_validation():
    pol = named_policies([{"kind": "twap"}])
    with pytest.raises(ValueError, match="reference"):
        BenchmarkSuite("x", "custom", "", pol, (Scenario(),), reference="RL")
    with pytest.raises(ValueError, match="category"):
        BenchmarkSuite("x", "fun", "", pol, (Scenario(),))
    with pytest.raises(ValueError, match="two policies"):
        named_policies([{"kind": "twap"}, {"kind": "twap"}])


def test_smoke_suite_runs_and_is_deterministic():
    s = get_suite("smoke")
    a, b = run_suite(s, n_episodes=60), run_suite(s, n_episodes=60)
    assert [r["mean_cv"] for r in a.rows] == [r["mean_cv"] for r in b.rows]
    assert len(a.rows) == 2 * 3 and len(a.comparisons) == 2 * 2
    c = a.comparisons[0]
    assert c["b"] == "TWAP" and c["design"] == "paired" and c["verdict"]
    assert a.label == "POST-STUDY / EXPLORATORY"


def test_comparison_estimates_are_paired_means():
    s = get_suite("smoke")
    r = run_suite(s, n_episodes=80)
    for c in r.comparisons:
        a = r.episodes[f"{c['scenario']}|{c['a']}|IScv"]
        b = r.episodes[f"{c['scenario']}|{c['b']}|IScv"]
        assert c["estimate"] == pytest.approx(float(np.mean(a - b)))


def test_missing_checkpoints_are_skipped_not_fatal(tmp_path):
    s = get_suite("liquidity")
    r = run_suite(s, n_episodes=30, models_dir=tmp_path)
    assert "RL" in r.skipped and {x["policy"] for x in r.rows} == {"TWAP", "AC", "ADAPT"}
    with pytest.raises(FileNotFoundError):
        run_suite(s, n_episodes=30, models_dir=tmp_path, skip_unavailable=False)


def test_rl_suite_runs_with_any_checkpoints(tiny_models):
    r = run_suite(get_suite("gate"), n_episodes=40, models_dir=tiny_models, seeds=(0,))
    # the tiny fixture has 2 members; the gate suite expects 5 -> RL is skipped cleanly
    assert "RL" in r.skipped


def test_record_run_writes_a_complete_run(tmp_path):
    store = FileArtifactStore(tmp_path)
    s = get_suite("smoke")
    res = run_suite(s, n_episodes=40)
    d = record_run(res, name="smoke", config=res.suite, config_hash=hash_config(res.suite),
                   store=store)
    names = {p.name for p in d.rglob("*")}
    assert {"manifest.json", "results.json", "episodes.npz", "REPORT.md",
            "comparison_vs_reference.png"} <= names
    man = json.loads((d / "manifest.json").read_text())
    assert man["label"] == "POST-STUDY / EXPLORATORY" and man["kind"] == "benchmark"
    assert set(store.verify(d.name).values()) == {"ok"}
    ep = np.load(d / "episodes.npz")
    assert ep["ID/sqrt|TWAP|IS"].shape == (40,)
    assert "POST-STUDY" in (d / "REPORT.md").read_text()
