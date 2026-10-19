"""Figures for a recorded benchmark run, generated from its ``results.json`` only.

Honesty rules applied to every figure here:

* paired differences are drawn against a zero line with the full 95% CI;
* the practical threshold (+/-0.25 bp unless the run declares another) is drawn as a band,
  so "significant but tiny" is visible as such;
* axes are linear, start where the data put them, and no scenario or policy is dropped;
* every figure is stamped POST-STUDY / EXPLORATORY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .style import COLOR, INK2, PALETTE, plt, setup

STAMP = "POST-STUDY / EXPLORATORY: not part of the pre-registered study"


def _color(policy: str, i: int) -> str:
    base = policy.split("_r")[0]
    if base in COLOR:
        return COLOR[base]
    if policy.startswith("GATE_"):
        return {"GATE_ood": COLOR["A_OOD"], "GATE_ensemble": COLOR["A_ENS"],
                "GATE_shadow": COLOR["A_SHADOW"]}.get(policy, PALETTE[i % len(PALETTE)])
    return PALETTE[i % len(PALETTE)]


def _stamp(ax) -> None:
    """Label above the plot area, clear of tick labels; titles are left-aligned."""
    title = ax.get_title()
    ax.set_title("")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.005, STAMP, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK2)


def comparison_figure(res: dict[str, Any], out: Path) -> Path | None:
    comps = res["comparisons"]
    if not comps:
        return None
    scen = list(dict.fromkeys(c["scenario"] for c in comps))
    pols = list(dict.fromkeys(c["a"] for c in comps))
    ref = comps[0]["b"]
    thr = comps[0]["threshold_bp"]
    h = max(2.4, 0.32 * len(scen) * max(1, len(pols)) ** 0.5 + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, h))
    step = 0.8 / max(1, len(pols))
    for j, p in enumerate(pols):
        for i, s in enumerate(scen):
            c = next((x for x in comps if x["scenario"] == s and x["a"] == p), None)
            if c is None:
                continue
            y = i + (j - (len(pols) - 1) / 2) * step
            ax.errorbar(c["estimate"], y, xerr=[[c["estimate"] - c["ci_low"]],
                                                [c["ci_high"] - c["estimate"]]],
                        fmt="o", color=_color(p, j), ms=5, elinewidth=1.6, capsize=0,
                        label=p if i == 0 else None)
    ax.axvspan(-thr, thr, color="#e4e3df", alpha=0.6, lw=0,
               label=f"±{thr:g} bp practical threshold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(scen)), scen)
    ax.invert_yaxis()
    ax.set(xlabel=f"paired difference in expected IS vs {ref} (bp); < 0 = cheaper than {ref}",
           title=f"{res['suite']['name']}: policy − {ref}, 95% paired-bootstrap CI")
    ax.legend(fontsize=7, loc="best")
    _stamp(ax)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "comparison_vs_reference.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def metric_figure(res: dict[str, Any], metric: str, out: Path) -> Path | None:
    rows = [r for r in res["rows"] if metric in r["metrics"]]
    if not rows:
        return None
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(scen) * len(pols) ** 0.5 + 2), 3.2))
    w = 0.8 / len(pols)
    x = np.arange(len(scen))
    for j, p in enumerate(pols):
        vals = [next((r["metrics"][metric] for r in rows
                      if r["scenario"] == s and r["policy"] == p), np.nan) for s in scen]
        ax.bar(x + (j - (len(pols) - 1) / 2) * w, vals, width=w * 0.92, color=_color(p, j),
               label=p)
    ax.set_xticks(x, scen, rotation=45 if len(scen) > 4 else 0,
                  ha="right" if len(scen) > 4 else "center")
    ax.set(ylabel=f"{metric} of realised IS (bp)", title=f"{res['suite']['name']}: {metric}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.legend(fontsize=7, ncols=min(4, len(pols)))
    _stamp(ax)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"metric_{metric}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fallback_figure(res: dict[str, Any], out: Path) -> Path | None:
    rows = [r for r in res["rows"] if "fallback_rate" in r]
    if not rows:
        return None
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    M = np.array([[next((r["fallback_rate"] for r in rows
                         if r["scenario"] == s and r["policy"] == p), np.nan) for s in scen]
                  for p in pols])
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(scen) + 2), 0.45 * len(pols) + 1.4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(scen)), scen, rotation=45, ha="right")
    ax.set_yticks(range(len(pols)), pols)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of orders sent to fallback")
    ax.set_title(f"{res['suite']['name']}: gate intervention frequency")
    _stamp(ax)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "gate_intervention_frequency.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def make_benchmark_figures(res: dict[str, Any], out: Path) -> list[Path]:
    setup()
    out = Path(out)
    made = [comparison_figure(res, out), fallback_figure(res, out)]
    for m in ("mean", "cvar95"):
        made.append(metric_figure(res, m, out))
    return [p for p in made if p is not None]


__all__ = ["make_benchmark_figures"]
