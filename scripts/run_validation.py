"""Stage 4: evaluate every policy and compute gate features on the VALIDATION split.

Validation data are used to fit all gates / controls (and, during development, to debug the
pipeline by splitting validation in halves).  Writes results/raw/val_<impact>.pkl.
"""

import argparse

from adexec.config import IMPACT_MODELS, MIXTURE_WEIGHTS
from adexec.experiment import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact", nargs="+", default=list(IMPACT_MODELS))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    for im in a.impact:
        print("validation", im, "->", evaluate("val", im, list(MIXTURE_WEIGHTS),
                                                 workers=a.workers), flush=True)
