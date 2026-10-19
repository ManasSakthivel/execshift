"""Provenance of the pre-registered study's published numbers (read-only).

The confirmatory study predates the experiment registry, so it has no run manifest. This
module reconstructs one from the frozen artifacts and never writes to them:

* ``study_manifest()``: a registry-schema manifest for the study (lock, frozen-file
  hashes, the git commit that last changed ``results/authoritative.json``, seeds, episode
  counts, recorded environment, artifact hashes).
* ``explain(claim)``: for a headline claim (``H1`` ... ``H6``), the number, its JSON
  pointer inside ``results/authoritative.json``, the code that computed it, the data split
  and seeds, and the pre-registered decision rule.

Both answer "what exactly generated this number?" for the study without re-running it.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, cast

from .. import paths
from ..config import (
    ALPHA,
    CONDITIONS,
    ENSEMBLE_SIZE,
    IMPACT_MODELS,
    MIXTURE_WEIGHTS,
    N_EPISODES,
    N_REPLICATES,
    PRIMARY_IMPACT,
    SPLIT_TAGS,
)
from .registry import MANIFEST_KEYS, SCHEMA_VERSION, sha256_file

STUDY_ID = "preregistered-study-v1"
STUDY_LABEL = "PRE-REGISTERED CONFIRMATORY (frozen; see research/prereg.lock)"
STUDY_ARTIFACTS = ("results/authoritative.json", "results/authoritative.reference.json",
                   "results/figure_data.json", "results/tables.md", "results/rl_manifest.json",
                   "results/simulator_validation.json", "results/baselines_sqrt.json",
                   "results/baselines_linear.json", "results/baselines_transient.json",
                   "results/test_access_log.jsonl", "research/PREREGISTRATION.md",
                   "research/prereg.lock")

# claim -> (JSON pointer in authoritative.json, generating code, decision rule)
CLAIMS: dict[str, dict[str, str]] = {
    "H1": {"pointer": "/primary/H1/comparisons/RL-TWAP",
           "code": "adexec/analysis.py::h1 -> compare(RL, FB, {ID: 1}) -> "
                   "stats.crossed_bootstrap",
           "rule": "RL - TWAP and RL - AC both significant (Holm) and below -0.25 bp"},
    "H2": {"pointer": "/primary/H2/conditions",
           "code": "adexec/analysis.py::h2 (difference-in-differences vs ID, Holm over 12)",
           "rule": "at least one shift erodes RL's edge by more than 0.25 bp (significant)"},
    "H3": {"pointer": "/primary/H3/scores/RCG",
           "code": "adexec/analysis.py::h3 (Spearman, AUROC per replicate, crossed bootstrap "
                   "CI, B=1000, seed 7)",
           "rule": "lower 95% CI bound of Spearman rho > 0 and of AUROC(regret > 0) > 0.5"},
    "H4": {"pointer": "/primary/H45/H4/comparison",
           "code": "adexec/analysis.py::h45 -> compare(RCG, RL, MIXTURE_WEIGHTS)",
           "rule": "RCG - RL significant and below -0.25 bp (pre-registered SESOI)"},
    "H5": {"pointer": "/primary/H45/H5/comparisons",
           "code": "adexec/analysis.py::h45 (RCG vs A3_BLEND, A9_RULE, A10_RAND, FB; Holm)",
           "rule": "RCG significantly cheaper than every control"},
    "H6": {"pointer": "/impact_models, /latency",
           "code": "adexec/reporting/study_report.py::verdicts (impact: H3, H4, H5 decisions "
                   "under linear and transient; latency: H5 decision at 0, 1, 5 s)",
           "rule": "every primary conclusion keeps its sign and significance under each "
                   "alternative impact model and latency"},
}


def _pointer(doc: Any, pointer: str) -> Any:
    cur = doc
    for part in pointer.strip("/").split("/"):
        cur = cur[part]
    return cur


def _git_last_commit(rel: str) -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=paths.ROOT,
                             capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def load_authoritative() -> dict[str, Any]:
    p = paths.RESULTS / "authoritative.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; it is committed with the repository")
    return json.loads(p.read_text())


def study_config() -> dict[str, Any]:
    """The study design as data, read from the frozen ``adexec/config.py``."""
    from ..analysis import SESOI
    from ..market import STUDY_LATENCY_GRID_S
    from ..rl import PPO_KWARGS, TOTAL_STEPS
    from ..stats import B_BOOT, BOOT_SEED

    pk = cast(dict[str, Any], PPO_KWARGS["policy_kwargs"])

    return {
        "conditions": [{"name": c.name, "family": c.family, "severity": c.severity}
                       for c in CONDITIONS],
        "mixture_weights": MIXTURE_WEIGHTS, "split_tags": SPLIT_TAGS,
        "episodes": N_EPISODES, "replicates": N_REPLICATES, "ensemble_size": ENSEMBLE_SIZE,
        "impact_models": list(IMPACT_MODELS), "primary_impact": PRIMARY_IMPACT,
        "latency_grid_s": list(STUDY_LATENCY_GRID_S), "alpha": ALPHA, "sesoi_bp": SESOI,
        "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "design": "crossed replicate x episode"},
        "ppo": {"total_steps": TOTAL_STEPS, **{k: v for k, v in PPO_KWARGS.items()
                                                 if k != "policy_kwargs"},
                "net_arch": pk["net_arch"], "log_std_init": pk["log_std_init"]},
        "seeds": {"episode": "SeedSequence([split_tag, condition_index, episode])",
                  "ppo_member": "replicate * 5 + member",
                  "ppo_stream": "rltrain tag 44, stream = impact_index * 1000 + member seed"},
    }


def study_manifest() -> dict[str, Any]:
    from ..guard import FROZEN, check_lock

    a = load_authoritative()
    lock = json.loads((paths.RESEARCH / "prereg.lock").read_text())
    try:
        check_lock()
        lock_status = "OK: frozen files match research/prereg.lock"
    except Exception as e:  # ContaminationError or a missing file
        lock_status = f"MISMATCH: {e}"
    status = json.loads((paths.RESULTS / "status.json").read_text()) \
        if (paths.RESULTS / "status.json").exists() else {}
    arts = {rel: sha256_file(paths.ROOT / rel) for rel in STUDY_ARTIFACTS
            if (paths.ROOT / rel).exists()}
    p = a["primary"]
    man = {
        "experiment_id": STUDY_ID, "label": STUDY_LABEL, "kind": "confirmatory",
        "created_utc": lock["locked_at"],
        "git_commit": _git_last_commit("results/authoritative.json"), "git_dirty": None,
        "config_hash": lock["preregistration_sha256"], "config": study_config(),
        "seeds": {"replicates": a["primary"]["replicates"],
                  "bootstrap_seed": study_config()["bootstrap"]["seed"]},
        "policies": ["TWAP (FB)", "AC (tuned)", "ADAPT (tuned)", "RL1", "RL (5-member)",
                     "BLEND_0.25/0.5/0.75", "RCG", "A9_RULE", "A_OOD", "A_ENS", "A_SHADOW",
                     "A10_RAND", "ORACLE"],
        "environment": {"recorded": status.get("hardware")},
        "impact_model": PRIMARY_IMPACT,
        "scenarios": [c.name for c in CONDITIONS],
        "metrics": {"H4_rcg_minus_rl_bp": p["H45"]["H4"]["comparison"]["estimate"],
                    "H3_rcg_spearman": p["H3"]["scores"]["RCG"]["spearman"],
                    "H3_rcg_auroc": p["H3"]["scores"]["RCG"]["auroc_harm"]},
        "artifact_paths": arts,
        "software_version": "study code frozen at the lock (see frozen_files)",
        "prereg_lock_sha256": sha256_file(paths.RESEARCH / "prereg.lock"),
        "schema_version": SCHEMA_VERSION,
        "extra": {"lock_status": lock_status, "frozen_files": lock["files"],
                  "frozen_list": list(FROZEN),
                  "prereg_sha256": a.get("prereg_sha256")},
    }
    assert all(k in man for k in MANIFEST_KEYS)
    return man


def explain(claim: str) -> dict[str, Any]:
    """Provenance of one headline claim of the study."""
    from ..reporting.study_report import verdicts

    key = claim.upper()
    if key not in CLAIMS:
        raise KeyError(f"unknown claim {claim!r}; expected one of {sorted(CLAIMS)}")
    a = load_authoritative()
    c = CLAIMS[key]
    if key == "H6":
        v = verdicts(a)
        value: Any = {"impact": v["H6_impact"], "latency": v["H6_latency"]}
    else:
        value = _pointer(a, c["pointer"])
    if key in ("H1", "H4"):
        value = {k: value[k] for k in ("estimate", "ci_low", "ci_high", "p_value", "d_z")}
    elif key == "H3":
        value = {k: value[k] for k in ("spearman", "auroc_harm", "auprc_harm",
                                       "auprc_harm_base", "ci")}
    elif key == "H2":
        value = {cond: {k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_holm")}
                 for cond, r in value.items()}
    elif key == "H5":
        value = {k: {q: r[q] for q in ("estimate", "ci_low", "ci_high", "p_holm")}
                 for k, r in value.items()}
    return {"claim": key, "verdict": verdicts(a)[key], "value": value,
            "units": "bp of arrival notional (IS); rho and AUROC are unitless",
            "source": "results/authoritative.json" + (c["pointer"] if key != "H6" else ""),
            "computed_by": c["code"], "decision_rule": c["rule"],
            "data": "TEST split (tag 33), 10 replicates x 4,000 ID + 12 x 1,000 shifted "
                    "episodes; gates fitted on VALIDATION (tag 22) under leave-one-family-out",
            "estimator": "control-variate IS (IScv); realised-IS sensitivity in 'realised'",
            "prereg_sha256": a.get("prereg_sha256"), "locked_at": a.get("locked_at"),
            "results_commit": _git_last_commit("results/authoritative.json")}


__all__ = ["CLAIMS", "STUDY_ID", "explain", "load_authoritative", "study_config",
           "study_manifest"]
