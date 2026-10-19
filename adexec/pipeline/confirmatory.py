"""Stage 5: the confirmatory run. Touches the TEST split exactly once per configuration.

Requires the pre-registration lock (research/prereg.lock). Produces results/raw/test_*.pkl
and results/authoritative.json.

This is the ONLY module outside ``adexec/guard.py`` allowed to request the test split
(enforced statically by ``tests/test_integrity.py``).
"""

from __future__ import annotations

import argparse
import json
import os

from .. import paths
from ..analysis import analyze
from ..config import IMPACT_MODELS, MIXTURE_WEIGHTS, PRIMARY_IMPACT
from ..guard import check_lock
from ..market import STUDY_LATENCY_GRID_S
from .evaluation import evaluate, load

LATENCIES = STUDY_LATENCY_GRID_S  # pre-registered robustness grid (seconds): 0, 1, 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    lock = check_lock()
    os.environ["ADEXEC_CONFIRMATORY"] = "1"
    conds = list(MIXTURE_WEIGHTS)
    if not a.analyze_only:
        for im in IMPACT_MODELS:
            evaluate("test", im, conds, workers=a.workers)
        for lat in LATENCIES:
            evaluate("test", PRIMARY_IMPACT, conds, latency_s=lat, tag=f"_lat{lat:g}",
                     workers=a.workers)
    raw = paths.raw_dir()
    out = {"prereg_sha256": lock["preregistration_sha256"], "locked_at": lock["locked_at"],
           "synthetic": True}
    val = {im: load(raw / f"val_{im}.pkl") for im in IMPACT_MODELS}
    out["primary"] = analyze(val[PRIMARY_IMPACT], load(raw / f"test_{PRIMARY_IMPACT}.pkl"))
    out["impact_models"] = {im: analyze(val[im], load(raw / f"test_{im}.pkl"))
                            for im in IMPACT_MODELS if im != PRIMARY_IMPACT}
    out["latency"] = {f"{lat:g}": analyze(val[PRIMARY_IMPACT],
                                          load(raw / f"test_{PRIMARY_IMPACT}_lat{lat:g}.pkl"),
                                          full=False)
                      for lat in LATENCIES}
    (paths.RESULTS / "authoritative.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n")
    print("wrote results/authoritative.json")
    return 0
