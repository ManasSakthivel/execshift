"""Scientific-integrity tests: these fail if the study's safeguards are broken."""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

from adexec import guard
from adexec.config import (
    CONDITIONS,
    MIXTURE_WEIGHTS,
    N_EPISODES,
    N_REPLICATES,
    SHIFTS,
    SPLIT_TAGS,
    SYNTHETIC,
)
from adexec.sim import generate_market

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
AUTH = RESULTS / "authoritative.json"

# The only modules allowed to name the test split: the frozen config that defines it (split
# tag and episode counts), the guard that refuses it, and the confirmatory stage that
# requests it.  Every other module in the package and in scripts/
# (including all post-study platform code) is checked statically.
TEST_SPLIT_ALLOWED = {"adexec/config.py", "adexec/guard.py", "adexec/pipeline/confirmatory.py"}
DEV_CODE = sorted(
    str(p.relative_to(ROOT)) for p in [*(ROOT / "adexec").rglob("*.py"),
                                       *(ROOT / "scripts").glob("*.py")]
    if str(p.relative_to(ROOT)) not in TEST_SPLIT_ALLOWED)


def _calls_test_split(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "test":
            return True
    return False


def test_static_scan_covers_the_codebase():
    assert len(DEV_CODE) > 40  # guards against the glob silently matching nothing
    assert "adexec/tuning.py" in DEV_CODE and "adexec/benchmarks/runner.py" in DEV_CODE


@pytest.mark.parametrize("rel", DEV_CODE)
def test_development_code_never_requests_test_split(rel):
    p = ROOT / rel
    if p.exists():
        assert not _calls_test_split(p), f"{rel} references the test split"


def test_test_split_refused_outside_confirmatory(monkeypatch):
    monkeypatch.delenv("ADEXEC_CONFIRMATORY", raising=False)
    with pytest.raises(guard.ContaminationError):
        guard.split_rngs("test", "ID", 3)


def test_split_streams_are_disjoint():
    tags = list(SPLIT_TAGS.values())
    assert len(set(tags)) == len(tags)
    n = 50
    draws = {}
    for split in ("train", "val", "rltrain", "simval"):
        m = generate_market(CONDITIONS[0].params, guard.episode_rngs(SPLIT_TAGS[split], 0,
                                                                    np.arange(n)))
        draws[split] = {tuple(np.round(m.eps[i, 1:6], 12)) for i in range(n)}
    keys = list(draws)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not draws[a] & draws[b], f"{a} and {b} share episodes"


def test_baseline_tuning_uses_train_split_only():
    for f in RESULTS.glob("baselines_*.json"):
        r = json.loads(f.read_text())
        assert r["split"] == "train" and r["split_tag"] == SPLIT_TAGS["train"]
    src = (ROOT / "adexec/tuning.py").read_text()
    assert 'split_rngs("train"' in src and '"val"' not in src


def test_gate_thresholds_fitted_on_validation_not_test():
    src = (ROOT / "adexec/analysis.py").read_text()
    # gates are fitted on `fit_raw` (validation) and only *applied* to eval_raw
    assert "GateSuite(rf[\"train_F\"]" in src and "apply_gates(suite, rf[\"conds\"]" in src
    gsrc = (ROOT / "adexec/gates.py").read_text()
    assert "test" not in re.findall(r"split_rngs\((\"\w+\")", gsrc)


def test_mixture_weights_sum_to_one_and_cover_all_conditions():
    assert abs(sum(MIXTURE_WEIGHTS.values()) - 1) < 1e-12
    assert set(MIXTURE_WEIGHTS) == {c.name for c in CONDITIONS}
    assert len(SHIFTS) == 12


def test_random_switching_budget_is_matched():
    from adexec.gates import GateSuite, apply_gates

    rng = np.random.default_rng(0)
    nf = 15

    def cond(n):
        IS = {k: rng.normal(5, 1, n) for k in ("RL", "FB", "BLEND_0.25", "BLEND_0.5",
                                                "BLEND_0.75")}
        return {"F": rng.normal(0, 1, (n, nf)), "IS": IS, "IScv": IS}

    val = {c.name: cond(300) for c in CONDITIONS}
    test = {c.name: cond(200) for c in CONDITIONS}
    out = apply_gates(GateSuite(rng.normal(0, 1, (500, nf))), val, test)
    for c, r in out.items():
        p = r["fallback"]["RCG"].mean()
        assert r["meta"]["A10_p"] == pytest.approx(p)
        exp = (1 - p) * test[c]["IScv"]["RL"] + p * test[c]["IScv"]["FB"]
        assert np.allclose(r["IScv"]["A10_RAND"], exp)
        # random gate's expected fallback count equals RCG's realised fallback count
        assert np.isclose(np.sum((r["IScv"]["A10_RAND"] - test[c]["IScv"]["RL"])
                                 / np.where(test[c]["IScv"]["FB"] != test[c]["IScv"]["RL"],
                                            test[c]["IScv"]["FB"] - test[c]["IScv"]["RL"], 1)),
                          p * 200)


def test_crossed_bootstrap_uses_replicates_and_episodes_as_units():
    from adexec.stats import crossed_bootstrap

    rng = np.random.default_rng(1)
    R, E = 10, 400
    rep_effect = rng.normal(0, 1.0, (R, 1))  # large between-replicate variance
    d = {"ID": rep_effect + rng.normal(0, 1, (R, E))}
    both = crossed_bootstrap(d, {"ID": 1.0}, B=2000)
    eps_only = crossed_bootstrap(d, {"ID": 1.0}, B=2000, resample_replicates=False)
    # ignoring the replicate unit would understate the uncertainty badly
    assert (both["ci_high"] - both["ci_low"]) > 3 * (eps_only["ci_high"] - eps_only["ci_low"])
    sd_true = np.sqrt(rep_effect.var(ddof=1) / R + 1 / (R * E))
    assert (both["ci_high"] - both["ci_low"]) / (2 * 1.96) == pytest.approx(sd_true, rel=0.35)


def test_everything_is_labelled_synthetic():
    assert SYNTHETIC is True
    for doc in ("README.md", "docs/FINAL_RESEARCH_REPORT.md"):
        p = ROOT / doc
        if p.exists():
            assert "synthetic" in p.read_text().lower(), f"{doc} must state data are synthetic"


@pytest.mark.skipif(not AUTH.exists(), reason="authoritative results not generated yet")
class TestAuthoritativeResults:
    def setup_method(self):
        self.a = json.loads(AUTH.read_text())

    def test_every_replicate_present(self):
        prim = self.a["primary"]
        assert prim["n_replicates"] == N_REPLICATES
        assert sorted(prim["replicates"]) == list(range(N_REPLICATES))
        for c, n in prim["n_episodes"].items():
            assert n == N_EPISODES["test"][c]
        per_rep = prim["H1"]["describe"]["RL"]["per_replicate_mean_cv"]
        assert len(per_rep) == N_REPLICATES and all(np.isfinite(per_rep))

    def test_no_best_run_reporting(self):
        # the primary RL numbers are means over replicates, not the best replicate
        d = self.a["primary"]["H1"]["describe"]["RL"]
        assert d["mean_cv"] == pytest.approx(np.mean(d["per_replicate_mean_cv"]), abs=1e-3)

    def test_prereg_lock_matches(self):
        lock = guard.check_lock()
        assert lock["preregistration_sha256"] == self.a["prereg_sha256"]

    def test_test_split_accessed_only_after_lock(self):
        log = RESULTS / "test_access_log.jsonl"
        assert log.exists()
        lock = json.loads(guard.LOCK.read_text())
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            assert rec["prereg_sha256"] == lock["preregistration_sha256"]
            assert rec["time"] >= lock["locked_at"]

    def test_report_and_tables_match_authoritative_numbers(self):
        from adexec.reporting.study_report import render_report

        report = (ROOT / "docs/FINAL_RESEARCH_REPORT.md").read_text()
        assert report == render_report(self.a), "report is stale: run `make report`"
        tables = ROOT / "results" / "tables.md"
        from adexec.reporting.tables import render_tables

        assert tables.read_text() == render_tables(self.a), "tables are stale"

    def test_readme_headline_numbers_match(self):
        from adexec.reporting.study_report import headline_numbers

        readme = (ROOT / "README.md").read_text()
        for s in headline_numbers(self.a):
            assert s in readme, f"README is missing / has stale headline value: {s}"

    @pytest.mark.parametrize("doc", ["README.md", "docs/RESULTS.md", "docs/FAQ.md"])
    def test_hand_written_docs_quote_only_generated_intervals(self, doc):
        # every "est [lo, hi]" in a hand-written doc must appear verbatim in the generated tables
        tables = (RESULTS / "tables.md").read_text()
        quoted = re.findall(r"[-+]?\d+\.\d+ \[[-+]?\d+\.\d+, [-+]?\d+\.\d+\]",
                            (ROOT / doc).read_text())
        assert quoted, f"{doc} quotes no intervals; is the pattern still right?"
        stale = [q for q in quoted if q not in tables]
        assert not stale, f"{doc} quotes intervals not in results/tables.md: {stale}"
