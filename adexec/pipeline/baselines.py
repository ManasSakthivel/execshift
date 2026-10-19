"""Stage 2: tune AC and ADAPT on the TRAIN split for each impact model
-> results/baselines_<impact>.json."""

from __future__ import annotations

import argparse
import json

from .. import paths
from ..config import IMPACT_MODELS
from ..tuning import tune_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impact", nargs="+", default=list(IMPACT_MODELS))
    for im in ap.parse_args(argv).impact:
        res = tune_all(im)
        paths.RESULTS.mkdir(exist_ok=True)
        (paths.RESULTS / f"baselines_{im}.json").write_text(json.dumps(res, indent=1))
        print(im, "AC kappa", res["AC"]["kappa"], "ADAPT", res["ADAPT"]["weights"])
    return 0
