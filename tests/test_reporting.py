"""Reporting: figures and reports are generated from artifacts and are labelled."""

import json

import pytest

from adexec import paths
from adexec.benchmarks import get_suite, run_suite
from adexec.reporting import supplementary
from adexec.reporting.benchmark_figures import make_benchmark_figures
from adexec.reporting.benchmark_report import render_benchmark_report


def test_benchmark_figures_and_report_from_results(tmp_path):
    res = run_suite(get_suite("smoke"), n_episodes=40).to_dict()
    figs = make_benchmark_figures(res, tmp_path)
    assert {f.name for f in figs} == {"comparison_vs_reference.png", "metric_mean.png",
                                      "metric_cvar95.png"}
    assert all(f.stat().st_size > 10_000 for f in figs)
    md = render_benchmark_report(res, "run-x")
    assert "POST-STUDY / EXPLORATORY" in md and "| verdict |" in md
    assert md.count("| ID/sqrt |") == 3 + 2  # 3 metric rows + 2 comparison rows


def test_supplementary_figures_from_committed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FIGURES", tmp_path)
    a = json.loads((paths.RESULTS / "authoritative.json").read_text())
    for fn in (supplementary.s1_policy_selection, supplementary.s2_gate_interventions,
               supplementary.s3_impact_sensitivity):
        assert fn(a).exists()


def test_supplementary_data_is_labelled_and_consistent_with_the_study():
    sd = json.loads((paths.RESULTS / supplementary.SUPP_JSON).read_text())
    assert "POST-STUDY DESCRIPTIVE" in sd["label"]
    a = json.loads((paths.RESULTS / "authoritative.json").read_text())
    rl_id = a["primary"]["H1"]["describe"]["RL"]
    assert abs(sd["tail_risk"]["ID"]["RL"]["cvar95"] - rl_id["cvar95"]) < 1e-9
    h = sd["score_regret_hist2d"]["RCG"]
    assert sum(map(sum, h["counts"])) == h["n"] == 160_000  # no episode dropped
    # S7's ROC (sklearn) must reproduce the study's rank-based AUROC (stats.auroc)
    for name, r in sd["roc_harm"].items():
        assert r["auroc"] == pytest.approx(a["primary"]["H3"]["scores"][name]["auroc_harm"],
                                           abs=1e-9)
        assert r["mean_tpr"][0] >= 0 and r["mean_tpr"][-1] == pytest.approx(1.0)
