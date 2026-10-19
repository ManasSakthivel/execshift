"""Stage 4: evaluate every policy and compute gate features on the VALIDATION split.

Validation data are used to fit all gates and controls. During development they were also
used to debug the pipeline, by splitting validation in halves. Writes
results/raw/val_<impact>.pkl.
"""

from __future__ import annotations

import argparse

from ..config import IMPACT_MODELS, MIXTURE_WEIGHTS
from .evaluation import evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impact", nargs="+", default=list(IMPACT_MODELS))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    for im in a.impact:
        print("validation", im, "->", evaluate("val", im, list(MIXTURE_WEIGHTS),
                                                 workers=a.workers), flush=True)
    return 0
