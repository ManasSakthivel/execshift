"""Reproducibility: frozen code, historical artifacts, determinism, regression pins."""

import hashlib
import json
import warnings
import zipfile

import numpy as np
import pytest

from adexec import guard
from adexec.config import CONDITION_BY_NAME
from adexec.paths import ROOT
from adexec.pipeline.verify import compare
from adexec.policies import TWAP, Adaptive, AlmgrenChriss
from adexec.sim import episode_rngs, generate_market, rollout

# SHA-256 of the committed study artifacts at the confirmatory run (commit 0e7f68c) and the
# lock (23ac727). A same-platform `make reproduce-paper` regenerates these byte for byte.
# If one changes, either a historical result was modified (not allowed) or a cross-platform
# reproduction overwrote it (compare with `execshift reproduce --verify` instead).
HISTORICAL = {
    "results/authoritative.json":
        "2259c1ba0ee143107bb76ca6594c3a81df6a7b18db7a8227001863f507723e67",
    "results/authoritative.reference.json":
        "2259c1ba0ee143107bb76ca6594c3a81df6a7b18db7a8227001863f507723e67",
    "results/figure_data.json":
        "056afab7c9b72613bd96175025ae69d72b11ee26e63ffb540290bec7864ac7b9",
    "results/tables.md": "255462297c4a4b39ac788d48023ecf2356b94eec16ac1f76d3c5d45c8b820d1a",
    "results/simulator_validation.json":
        "b319ef11b58b5255eba8f153b09d8135c06c012bddff658a2765e8eaaa972016",
    "results/baselines_sqrt.json":
        "aea693d6b1fc17d5bd5978766c1a0c77f9dc8a5aee082de78d057b257d0bad21",
    "results/baselines_linear.json":
        "442c1c1dbba0c527ad190f04349314c9bcbf3e6a2a5c50dfbe29d35afa994904",
    "results/baselines_transient.json":
        "55807f79539849faff69c93538d07abd0f186d3c370b76017c9bcc2df6ca921e",
    "research/PREREGISTRATION.md":
        "2abf320fda1f2d0487cb7ad5d450112f5017601e85da3b2b5ab997efa90aec46",
    "research/prereg.lock": "ec87ae3750ceba3857e0abcfabdea1591a1c6f7904ae122d092a6c7be84fc48d",
}


@pytest.mark.parametrize("rel", sorted(HISTORICAL))
def test_historical_artifacts_are_unchanged(rel):
    data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == HISTORICAL[rel], f"{rel} was modified"


def test_lock_covers_exactly_the_frozen_list_and_matches():
    lock = json.loads((ROOT / "research/prereg.lock").read_text())
    assert set(lock["files"]) == set(guard.FROZEN)
    guard.check_lock()


def test_frozen_files_resolve_to_the_imported_package():
    for rel in guard.FROZEN:
        assert guard.frozen_path(rel).exists()
    assert guard.frozen_path("adexec/sim.py").parent == guard.PACKAGE_DIR


def test_test_access_log_only_grows_after_the_lock():
    lock = json.loads((ROOT / "research/prereg.lock").read_text())
    lines = (ROOT / "results/test_access_log.jsonl").read_text().splitlines()
    assert len(lines) >= 780
    assert all(json.loads(x)["time"] >= lock["locked_at"] for x in lines)


# Regression pins: mean IS / IS_cv (bp) of three frozen policies on 300 simulator-validation
# episodes, computed with the frozen engine on 2026-10-03. Any engine change moves them.
PINS = [
    ("ID", "TWAP", 5.436870574398378, 6.698249899408524),
    ("ID", "AC", 5.5349052758383355, 6.718504878975097),
    ("ID", "ADAPT", 3.4016154358667783, 4.682246143309421),
    ("LIQ-sev", "TWAP", 15.11442101278058, 16.375703853027908),
    ("LIQ-sev", "AC", 15.208780098100757, 16.392270895424097),
    ("LIQ-sev", "ADAPT", 15.730214535306665, 17.252144212130194),
    ("LAT-sev", "TWAP", 5.424490660864752, 6.69367215155402),
    ("LAT-sev", "AC", 5.523785100185375, 6.714785095927195),
    ("LAT-sev", "ADAPT", 3.513952066901878, 4.794932089028395),
]
POLS = {"TWAP": TWAP(), "AC": AlmgrenChriss(1.0), "ADAPT": Adaptive(1, 0.5, 0.5, 0, 0)}


@pytest.mark.parametrize("cond,pol,is_mean,cv_mean", PINS)
def test_engine_regression_pins(cond, pol, is_mean, cv_mean):
    m = generate_market(CONDITION_BY_NAME[cond].params, episode_rngs(55, 990, np.arange(300)))
    r = rollout(m, POLS[pol])
    assert float(r["IS"].mean()) == pytest.approx(is_mean, rel=1e-9, abs=1e-9)
    assert float(r["IS_cv"].mean()) == pytest.approx(cv_mean, rel=1e-9, abs=1e-9)


def test_ppo_training_is_bit_for_bit_deterministic(tmp_path):
    from adexec.rl import train_member

    hashes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in ("a", "b"):
            p = train_member(1, 2, tmp_path / d, "sqrt", total_steps=960, n_envs=8)
            hashes.append(hashlib.sha256(zipfile.ZipFile(p).read("policy.pth")).hexdigest())
    assert hashes[0] == hashes[1]


def test_verifier_passes_identical_and_catches_differences():
    ref = json.loads((ROOT / "results/authoritative.json").read_text())
    r = compare(ref, ref)
    assert r["primary_failures"] == [] and r["primary_checked"] > 1000
    assert r["supplementary_identical"] == r["supplementary_compared"]
    bad = json.loads(json.dumps(ref))
    bad["primary"]["H45"]["H4"]["comparison"]["estimate"] += 1.0
    bad["primary"]["H45"]["H4"]["supported"] = True
    bad["primary"]["H1"]["comparisons"]["RL-TWAP"]["per_replicate"].pop()
    fails = compare(ref, bad)["primary_failures"]
    assert any("H4/comparison/estimate" in f for f in fails)
    assert any("decision differs" in f for f in fails)
    assert any("list length differs" in f for f in fails)
