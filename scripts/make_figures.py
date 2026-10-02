"""Figures 2-10 from results/authoritative.json and results/figure_data.json (no hand-edited
values).  Figure 1 (simulator validation) is produced by scripts/validate_simulator.py."""

import json
from pathlib import Path

import numpy as np

from adexec.config import MIXTURE_WEIGHTS
from adexec.plotstyle import COLOR, INK2, LABEL, plt, setup

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"


def _ci(ax, y, est, lo, hi, color, label=None, marker="o"):
    ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt=marker, color=color, ecolor=color,
                elinewidth=2, capsize=0, ms=6, label=label)


def _zero(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0, color=INK2, lw=0.8)


def fig2_in_distribution(a):
    h1 = a["primary"]["H1"]
    rows = [("RL ensemble − TWAP", h1["comparisons"]["RL-TWAP"], COLOR["RL"]),
            ("RL single member − TWAP", h1["secondary"]["RL1-TWAP"], COLOR["RL"]),
            ("ADAPT (tuned) − TWAP", h1["secondary"]["ADAPT-TWAP"], COLOR["ADAPT"]),
            ("AC (tuned) − TWAP", h1["secondary"]["AC-TWAP"], COLOR["AC"]),
            ("RL ensemble − AC", h1["comparisons"]["RL-AC"], COLOR["RL"]),
            ("RL ensemble − ADAPT", h1["comparisons"]["RL-ADAPT"], COLOR["RL"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (_lab, r, col) in enumerate(rows[::-1]):
        _ci(ax, i, r["estimate"], r["ci_low"], r["ci_high"], col)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    _zero(ax)
    ax.set(xlabel="paired difference in expected IS (bp); < 0 favours the first policy",
           title="In-distribution test episodes: 10 replicates × 4,000 episodes, 95% crossed-bootstrap CI")
    ax.title.set_fontsize(8.5)
    fig.savefig(FIG / "fig2_in_distribution.png")


def fig3_seeds(a):
    h1 = a["primary"]["H1"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (key, _lab) in enumerate((("RL-TWAP", "RL − TWAP"), ("RL-AC", "RL − AC"),
                                    ("RL-ADAPT", "RL − ADAPT"))):
        r = h1["comparisons"][key]
        pr = np.array(r["per_replicate"])
        jitter = np.linspace(-0.18, 0.18, pr.size)
        ax.scatter(pr, i + jitter, s=14, color=COLOR["RL"], alpha=0.55, zorder=3,
                   label="replicate mean" if i == 0 else None)
        _ci(ax, i, r["estimate"], r["ci_low"], r["ci_high"], INK2, "mean, 95% CI" if i == 0 else None,
            marker="D")
    ax.set_yticks(range(3), ["RL − TWAP", "RL − AC", "RL − ADAPT"])
    _zero(ax)
    ax.set(xlabel="paired difference in expected IS (bp), ID test",
           title="Every replicate shown: no best-run reporting")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "fig3_seed_robustness.png")


def fig4_shift(fd):
    conds = list(MIXTURE_WEIGHTS)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for _k, (pol, off) in enumerate((("RL", -0.15), ("ADAPT", 0.15))):
        for i, c in enumerate(conds[::-1]):
            e, lo, hi = fd["per_condition"][c][pol]
            _ci(ax, i + off, e, lo, hi, COLOR[pol], LABEL[pol] + " − TWAP" if i == 0 else None)
    ax.set_yticks(range(len(conds)), conds[::-1])
    _zero(ax)
    ax.set(xlabel="expected IS minus TWAP (bp); > 0: worse than the fallback",
           title="Distribution shift: RL's edge over TWAP by test condition")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "fig4_distribution_shift.png")


def fig5_uncertainty(a, fd):
    sc = a["primary"]["H3"]["scores"]
    names = list(sc)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, n in enumerate(names[::-1]):
        lo, hi = sc[n]["ci"]["spearman"]
        col = COLOR["RCG"] if n.startswith("RCG") else INK2
        _ci(ax[0], i, sc[n]["spearman"], lo, hi, col)
    ax[0].set_yticks(range(len(names)), names[::-1])
    _zero(ax[0])
    ax[0].set(xlabel="Spearman ρ(score, counterfactual regret), pooled test",
              title="(a) Does the score rank regret?")
    for n, col in (("RCG", COLOR["RCG"]), ("RCG_SEEN", COLOR["ADAPT"]), ("A_OOD", COLOR["A_OOD"]),
                   ("A_ENS", COLOR["AC"])):
        d = fd["score_deciles"][n]
        y = np.array([np.nan if v is None else v for v in d["mean_regret"]], dtype=float)
        e = np.array([np.nan if v is None else v for v in d["se"]], dtype=float)
        ax[1].errorbar(d["score_rank_decile"], y, yerr=1.96 * e, color=col, marker="o",
                       label=LABEL.get(n, n.replace("_", " ")), capsize=0)
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set(xlabel="score decile (1 = lowest predicted risk)",
              ylabel="mean regret of RL vs TWAP (bp)", title="(b) Regret by score decile")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_uncertainty_vs_regret.png")


def fig6_calibration(a):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    sets = [("sqrt (primary)", a["primary"]), *[(k, v) for k, v in a["impact_models"].items()]]
    for axi, (name, r) in zip(ax, sets, strict=True):
        c = r["H3"]["calibration"]
        lo = min(min(c["predicted"]), min(c["realised"]))
        hi = max(max(c["predicted"]), max(c["realised"]))
        axi.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1, label="perfect calibration")
        axi.plot(c["predicted"], c["realised"], marker="o", color=COLOR["RCG"],
                 label="RCG (decile means)")
        axi.axhline(0, color=INK2, lw=0.6)
        axi.axvline(0, color=INK2, lw=0.6)
        axi.set(xlabel="predicted regret (bp)", ylabel="realised regret (bp, CV)",
                title=f"Reliability, impact = {name}")
        axi.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_calibration.png")


def fig6_risk_coverage(fd):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for n, col in (("RCG", COLOR["RCG"]), ("RCG_SEEN", COLOR["ADAPT"]), ("A_OOD", COLOR["A_OOD"]),
                   ("A_ENS", COLOR["AC"])):
        d = fd["risk_coverage"][n]
        ax.plot(d["coverage"], d["total_regret"], color=col, label=LABEL.get(n, n))
    d = fd["risk_coverage"]["RCG"]
    end = d["total_regret"][-1]
    ax.plot([0, 1], [0, end], color=INK2, ls="--", lw=1.2, label="random order")
    ax.set(xlabel="coverage (fraction of orders where RL is trusted)",
           ylabel="regret vs always-TWAP (bp per order)",
           title="Risk-coverage: lower is better")
    ax.legend(fontsize=7.5)
    fig.savefig(FIG / "fig7_risk_coverage.png")


def _gate_rows(res):
    h = res["H45"]
    rows = [("RCG − always-RL", h["H4"]["comparison"])]
    rows += [(f"RCG − {LABEL.get(k.split('-', 1)[1], k)}", v)
             for k, v in h["H5"]["comparisons"].items()]
    return rows


def fig7_gates(a):
    rows = _gate_rows(a["primary"])
    ex = a["primary"]["H45"]["exploratory"]
    rows += [("RCG − OOD gate", ex["RCG-A_OOD"]), ("RCG − ensemble gate", ex["RCG-A_ENS"]),
             ("RCG − shadow gate", ex["RCG-A_SHADOW"]), ("RCG − ADAPT", ex["RCG-ADAPT"]),
             ("RCG(seen family) − RL", ex["RCG_SEEN-RL"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, (_lab, r) in enumerate(rows[::-1]):
        col = COLOR["RCG"] if i >= len(rows) - 5 else INK2
        _ci(ax, i, r["estimate"], r["ci_low"], r["ci_high"], col)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    _zero(ax)
    ax.set(xlabel="difference in deployment-mixture expected IS (bp); < 0 favours RCG",
           title="Gate vs controls (confirmatory: top 5; exploratory: rest)")
    fig.savefig(FIG / "fig8_gate_comparison.png")


def _robust(ax, results: dict, title: str):
    keys = [("H4", "RCG − RL"), ("RCG-A3_BLEND", "RCG − A3"), ("RCG-A9_RULE", "RCG − A9"),
            ("RCG-A10_RAND", "RCG − A10"), ("RCG-FB", "RCG − TWAP")]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"]
    n = len(results)
    for j, (name, res) in enumerate(results.items()):
        for i, (k, _lab) in enumerate(keys):
            r = res["H45"]["H4"]["comparison"] if k == "H4" else res["H45"]["H5"]["comparisons"][k]
            y = i + (j - (n - 1) / 2) * 0.18
            _ci(ax, y, r["estimate"], r["ci_low"], r["ci_high"], cols[j % len(cols)],
                name if i == 0 else None)
    ax.set_yticks(range(len(keys)), [k[1] for k in keys])
    _zero(ax)
    ax.set(xlabel="mixture expected IS difference (bp)", title=title)
    ax.legend(fontsize=7.5, loc="lower left")


def fig8_impact(a):
    res = {"sqrt (primary)": a["primary"], **a["impact_models"]}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    _robust(ax, res, "Robustness to the impact model (full pipeline re-run)")
    fig.savefig(FIG / "fig9_impact_robustness.png")


def fig9_latency(a):
    res = {"0.2 s (primary)": a["primary"], **{f"{k} s": v for k, v in a["latency"].items()}}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    _robust(ax, res, "Robustness to execution latency (all non-latency conditions)")
    fig.savefig(FIG / "fig10_latency_robustness.png")


def main():
    setup()
    a = json.loads((ROOT / "results" / "authoritative.json").read_text())
    fd = json.loads((ROOT / "results" / "figure_data.json").read_text())
    FIG.mkdir(exist_ok=True)
    fig2_in_distribution(a)
    fig3_seeds(a)
    fig4_shift(fd)
    fig5_uncertainty(a, fd)
    fig6_calibration(a)
    fig6_risk_coverage(fd)
    fig7_gates(a)
    fig8_impact(a)
    fig9_latency(a)
    print("figures written")


if __name__ == "__main__":
    main()
