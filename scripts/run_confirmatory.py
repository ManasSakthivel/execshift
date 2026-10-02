"""Stage 5: the confirmatory run.  Touches the TEST split exactly once per configuration.

Requires the pre-registration lock (research/prereg.lock).  Produces results/raw/test_*.pkl
and results/authoritative.json.
"""

import argparse
import json
import os

from adexec.analysis import analyze
from adexec.config import IMPACT_MODELS, MIXTURE_WEIGHTS, PRIMARY_IMPACT
from adexec.experiment import RAW, evaluate, load
from adexec.guard import ROOT, check_lock

LATENCIES = (0.0, 1.0, 5.0)  # pre-registered robustness grid (seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    lock = check_lock()
    os.environ["ADEXEC_CONFIRMATORY"] = "1"
    conds = list(MIXTURE_WEIGHTS)
    if not a.analyze_only:
        for im in IMPACT_MODELS:
            evaluate("test", im, conds, workers=a.workers)
        for lat in LATENCIES:
            evaluate("test", PRIMARY_IMPACT, conds, latency_s=lat, tag=f"_lat{lat:g}",
                     workers=a.workers)
    out = {"prereg_sha256": lock["preregistration_sha256"], "locked_at": lock["locked_at"],
           "synthetic": True}
    val = {im: load(RAW / f"val_{im}.pkl") for im in IMPACT_MODELS}
    out["primary"] = analyze(val[PRIMARY_IMPACT], load(RAW / f"test_{PRIMARY_IMPACT}.pkl"))
    out["impact_models"] = {im: analyze(val[im], load(RAW / f"test_{im}.pkl"))
                            for im in IMPACT_MODELS if im != PRIMARY_IMPACT}
    out["latency"] = {f"{lat:g}": analyze(val[PRIMARY_IMPACT],
                                          load(RAW / f"test_{PRIMARY_IMPACT}_lat{lat:g}.pkl"),
                                          full=False)
                      for lat in LATENCIES}
    (ROOT / "results" / "authoritative.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n")
    print("wrote results/authoritative.json")


if __name__ == "__main__":
    main()
