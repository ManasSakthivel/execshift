"""Tune AC and ADAPT on the TRAIN split for each impact model -> results/baselines_<im>.json."""

import argparse
import json
from pathlib import Path

from adexec.config import IMPACT_MODELS
from adexec.tuning import tune_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact", nargs="+", default=list(IMPACT_MODELS))
    for im in ap.parse_args().impact:
        res = tune_all(im)
        Path("results").mkdir(exist_ok=True)
        Path(f"results/baselines_{im}.json").write_text(json.dumps(res, indent=1))
        print(im, "AC kappa", res["AC"]["kappa"], "ADAPT", res["ADAPT"]["weights"])
