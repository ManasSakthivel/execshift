"""Configuration, hashing, manifests, artifact store and study provenance."""

import json
import re
from pathlib import Path

import pytest

from adexec.config import CONDITION_BY_NAME
from adexec.experiments.config import (
    REGIMES,
    ConfigError,
    config_from_dict,
    hash_config,
    load_config,
)
from adexec.experiments.registry import (
    MANIFEST_KEYS,
    FileArtifactStore,
    RunNotFoundError,
    build_manifest,
    environment_info,
    make_experiment_id,
)
from adexec.experiments.study import CLAIMS, explain, study_config, study_manifest
from adexec.paths import ROOT

EXAMPLES = sorted((ROOT / "configs" / "examples").glob("*.toml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_configs_parse_and_round_trip(path):
    cfg = load_config(path)
    again = config_from_dict(cfg.to_dict())
    assert again == cfg and again.config_hash == cfg.config_hash
    assert cfg.label == "POST-STUDY / EXPLORATORY"


def test_config_and_store_satisfy_interfaces(tmp_path):
    from adexec import interfaces as I

    cfg = load_config(EXAMPLES[0])
    assert isinstance(cfg, I.Experiment)
    assert isinstance(FileArtifactStore(tmp_path), I.ArtifactStore)


def test_hash_ignores_key_order_and_defaults():
    a = config_from_dict({"experiment": {"name": "x"}, "market": {"conditions": ["ID"]}})
    b = config_from_dict({"market": {"conditions": ["ID"], "impact_model": "sqrt"},
                          "experiment": {"kind": "evaluate", "name": "x"}})
    assert a.config_hash == b.config_hash
    c = config_from_dict({"experiment": {"name": "x"}, "evaluation": {"seeds": [1]}})
    assert c.config_hash != a.config_hash
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})


@pytest.mark.parametrize("bad,match", [
    ({"experiment": {"name": "x"}, "markt": {}}, "unknown key"),
    ({"experiment": {"name": "x"}, "market": {"conditon": ["ID"]}}, "unknown key"),
    ({"experiment": {}}, "name is required"),
    ({"experiment": {"name": "x"}, "market": {"conditions": ["VOL-x"]}}, "unknown condition"),
    ({"experiment": {"name": "x"}, "market": {"liquidity": "awful"}}, "liquidity"),
    ({"experiment": {"name": "x"}, "market": {"overrides": {"perm_coef": 0}}}, "may only"),
    ({"experiment": {"name": "x"}, "evaluation": {"metrics": ["sharpe"]}}, "unknown metric"),
    ({"experiment": {"name": "x"}, "evaluation": {"threshold_bp": 0.1}}, "threshold_source"),
    ({"experiment": {"name": "x", "label": "Pre-registered"}}, "pre-registered"),
    ({"experiment": {"name": "x", "kind": "deploy"}}, "kind"),
    ({"experiment": {"name": "x"}, "policies": [{"tuned": True}]}, "no 'kind'"),
])
def test_invalid_configs_are_rejected_with_a_reason(bad, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(bad)


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
    (tmp_path / "bad.toml").write_text("[experiment\nname=1")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(tmp_path / "bad.toml")


def test_named_regimes_reuse_frozen_shift_magnitudes():
    assert REGIMES["volatility"]["crisis"] == {"sigma_mult": 3.0}
    liq = CONDITION_BY_NAME["LIQ-sev"].params
    assert REGIMES["liquidity"]["drought"]["half_spread_mult"] == liq.half_spread_mult
    cfg = config_from_dict({"experiment": {"name": "x"},
                            "market": {"conditions": ["ID"], "liquidity": "drought"}})
    s = cfg.market.scenarios()[0]
    assert not s.is_preregistered
    assert s.params().eta_mult == liq.eta_mult and s.params().volume_mult == liq.volume_mult


# ----------------------------------------------------------------------------- registry


def _manifest(store, d, **kw):
    return build_manifest(experiment_id=d.name, label="POST-STUDY / EXPLORATORY",
                          kind="evaluate", config={"a": 1}, config_hash="ab" * 32, seeds=[0],
                          policies=["TWAP"], impact_model="sqrt", scenarios=["ID"],
                          metrics={"m": 1.0}, run_dir=d, **kw)


def test_store_create_list_resolve_verify(tmp_path):
    store = FileArtifactStore(tmp_path)
    eid = make_experiment_id("demo", "0123456789abcdef", "2026-10-03T10:00:00Z")
    assert eid == "demo-01234567-20261003T100000"
    d = store.create_run(eid)
    (d / "results.json").write_text("{}")
    store.write_manifest(d, _manifest(store, d))
    assert store.list_runs() == [eid] and store.resolve("demo-0") == eid
    assert store.verify(eid) == {"results.json": "ok"}
    (d / "results.json").write_text('{"tampered": 1}')
    assert store.verify(eid) == {"results.json": "modified"}
    with pytest.raises(FileExistsError):
        store.create_run(eid)
    with pytest.raises(RunNotFoundError):
        store.resolve("zzz")


def test_manifest_has_every_required_key_and_no_personal_paths(tmp_path):
    store = FileArtifactStore(tmp_path)
    d = store.create_run("x-1")
    man = _manifest(store, d)
    assert all(k in man for k in MANIFEST_KEYS)
    blob = json.dumps(man)
    assert str(Path.home()) not in blob and str(tmp_path) not in blob
    assert re.fullmatch(r"[0-9a-f]{40}", man["git_commit"] or "0" * 40)
    with pytest.raises(ValueError, match="missing"):
        store.write_manifest(d, {"experiment_id": "x"})


def test_environment_info_records_versions():
    env = environment_info()
    assert env["python"].count(".") == 2 and env["packages"]["numpy"]


# ------------------------------------------------------------------------- study provenance


def test_committed_study_design_matches_frozen_code():
    committed = json.loads((ROOT / "configs/study/preregistered.json").read_text())
    committed.pop("_comment")
    assert committed == json.loads(json.dumps(study_config()))


@pytest.mark.parametrize("claim", sorted(CLAIMS))
def test_every_claim_traces_to_its_source(claim):
    e = explain(claim)
    assert e["source"].startswith("results/authoritative.json")
    assert e["verdict"] in ("PASS", "FAIL") and e["computed_by"] and e["decision_rule"]


def test_explain_reports_the_published_headline_numbers():
    h4, h3 = explain("H4"), explain("H3")
    assert h4["verdict"] == "FAIL" and h3["verdict"] == "PASS"
    assert round(h4["value"]["estimate"], 3) == -0.013
    assert (round(h4["value"]["ci_low"], 3), round(h4["value"]["ci_high"], 3)) == (-0.024,
                                                                                    -0.004)
    assert round(h3["value"]["spearman"], 3) == 0.108
    assert round(h3["value"]["auroc_harm"], 3) == 0.542
    assert explain("H6")["verdict"] == "FAIL"


def test_study_manifest_reports_an_intact_lock():
    man = study_manifest()
    assert all(k in man for k in MANIFEST_KEYS)
    assert man["extra"]["lock_status"].startswith("OK")
    assert "PRE-REGISTERED" in man["label"]
