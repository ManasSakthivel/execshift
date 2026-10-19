"""Shared figure style.  Colours follow the entity (policy), never its rank, and come from a
CVD-validated categorical palette in fixed order."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# entity -> colour (stable across every figure)
COLOR = {
    "RL": PALETTE[0], "FB": PALETTE[1], "TWAP": PALETTE[1], "ADAPT": PALETTE[2],
    "AC": PALETTE[3], "RCG": PALETTE[6], "A3_BLEND": PALETTE[4], "A9_RULE": PALETTE[5],
    "A10_RAND": "#8c8b86", "A_OOD": PALETTE[7], "A_ENS": PALETTE[3], "A_SHADOW": PALETTE[2],
    "ORACLE": "#b5b4ae",
    "sqrt": PALETTE[0], "linear": PALETTE[1], "transient": PALETTE[2],
}
LABEL = {"RL": "RL (always)", "FB": "TWAP (always)", "TWAP": "TWAP", "ADAPT": "ADAPT (tuned)",
         "AC": "AC (tuned)", "RCG": "RCG (proposed)", "A3_BLEND": "A3 blend",
         "A9_RULE": "A9 rule gate", "A10_RAND": "A10 random (matched)",
         "A_OOD": "OOD gate", "A_ENS": "Ensemble gate", "A_SHADOW": "Shadow gate",
         "ORACLE": "Oracle (bound)"}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.facecolor": "white", "axes.axisbelow": True, "axes.facecolor": "white",
    })
