"""Supplementary study figures S1-S7 (POST-STUDY DESCRIPTIVE re-analysis).

These figures re-summarise the frozen confirmatory outputs. They run **no new hypothesis
tests**, change no number in ``results/authoritative.json``, and use no new data:

* S1, S2 and S3 read only committed artifacts (``results/authoritative.json``);
* S4, S5 and S6 need per-episode costs, which live in the git-ignored
  ``results/raw/*.pkl``. ``compute_supplementary_data`` reduces them once to
  ``results/post_study/supplementary_data.json`` (committed, small), and the figures read
  that file.

| Figure | Question | Source |
|---|---|---|
| S1 policy selection | Which policy is cheapest in which regime? | authoritative.json |
| S2 gate interventions | How often does each gate fall back, per condition? | authoritative.json |
| S3 impact sensitivity | Does RL's edge over TWAP depend on the impact model? | authoritative.json |
| S4 tail risk | CVaR95 of realised cost per condition: RL vs ADAPT vs TWAP | raw -> supplementary_data |
| S5 RL regret | Distribution of per-episode RL regret vs TWAP per condition | raw -> supplementary_data |
| S6 uncertainty vs regret | Every test episode: RCG score and OOD distance vs regret | raw -> supplementary_data |
| S7 harm-detection ROC | How well does each score rank harmful orders (regret > 0)? | raw -> supplementary_data |

Each figure is written as PNG (for GitHub) and PDF (vector).

S6 is drawn deliberately plainly: a 2-D histogram of *all* pooled episodes on linear axes,
with the full range of both variables, no outlier removal, and the study's Spearman rho
printed on the panel. The weak relationship (rho = +0.108) should look weak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..config import IMPACT_MODELS, MIXTURE_WEIGHTS, PRIMARY_IMPACT
from .style import COLOR, INK2, LABEL, plt, setup

STAMP = ("POST-STUDY DESCRIPTIVE re-analysis of the frozen confirmatory outputs "
         "(no new tests)")
CONDS = list(MIXTURE_WEIGHTS)
SUPP_JSON = "post_study/supplementary_data.json"
# S7 scores: name (as in authoritative.json H3) -> feature column; None = RCG predicted regret
ROC_SCORES = {"RCG": None, "OOD distance": "ood_maha", "Ensemble action std": "u_ens_s0",
              "Shadow regret": "shadow_regret"}


def _out() -> Path:
    d = paths.FIGURES / "supplementary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.005, STAMP, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK2)


def _save(fig, name: str) -> Path:
    p = _out() / name
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"), metadata={"CreationDate": None})  # byte-stable
    plt.close(fig)
    return p


# --------------------------------------------------------------------------------------
# Data reduction (needs results/raw)
# --------------------------------------------------------------------------------------


def compute_supplementary_data() -> dict[str, Any]:
    """Reduce the confirmatory per-episode outputs to the statistics S4-S7 plot."""
    from ..analysis import EST, describe, gated_results, stack
    from ..features import FEATURE_COLS
    from ..pipeline.evaluation import load

    raw = paths.raw_dir()
    for f in (f"val_{PRIMARY_IMPACT}.pkl", f"test_{PRIMARY_IMPACT}.pkl"):
        if not (raw / f).exists():
            raise FileNotFoundError(f"{raw / f} not found: S4-S7 need the confirmatory raw "
                                    "outputs (regenerate with `make reproduce-paper`)")
    gres = gated_results(load(raw / f"val_{PRIMARY_IMPACT}.pkl"),
                         load(raw / f"test_{PRIMARY_IMPACT}.pkl"))
    tail = {c: {p: {k: describe(stack(gres, c, p, "IS"))[k] for k in ("mean", "sd", "cvar95")}
                | {"p95": float(np.mean(np.quantile(stack(gres, c, p, "IS"), 0.95, axis=1)))}
                for p in ("RL", "ADAPT", "FB", "AC")} for c in CONDS}
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    regret = {}
    for c in CONDS:
        r = (stack(gres, c, "RL", EST) - stack(gres, c, "FB", EST)).ravel()
        regret[c] = {"quantiles": dict(zip(map(str, qs), np.quantile(r, qs).tolist(),
                                           strict=True)),
                     "mean": float(r.mean()), "share_positive": float((r > 0).mean()),
                     "n": int(r.size)}
    rg = np.concatenate([np.concatenate([g[c][EST]["RL"] - g[c][EST]["FB"] for c in CONDS])
                         for g in gres])
    j = FEATURE_COLS.index("ood_maha")
    scores = {"RCG": np.concatenate([np.concatenate([g[c]["score_RCG"] for c in CONDS])
                                     for g in gres]),
              "OOD distance": np.concatenate([np.concatenate([g[c]["F"][:, j] for c in CONDS])
                                              for g in gres])}
    hist = {}
    y_edges = np.linspace(rg.min(), rg.max(), 61)
    for name, s in scores.items():
        x_edges = np.linspace(s.min(), s.max(), 61)
        H, _, _ = np.histogram2d(s, rg, bins=[x_edges, y_edges])
        hist[name] = {"x_edges": x_edges.tolist(), "y_edges": y_edges.tolist(),
                      "counts": H.astype(int).tolist(), "n": int(s.size)}
    # ROC for detecting harm (regret > 0), per replicate as the study computes AUROC, with
    # sklearn's roc_curve: an implementation independent of stats.auroc.
    from sklearn.metrics import auc, roc_curve

    fpr_grid = np.linspace(0, 1, 101)
    roc = {}
    for name, col in ROC_SCORES.items():
        tprs, aucs = [], []
        for g in gres:
            s = np.concatenate([g[c]["score_RCG"] if col is None
                                else g[c]["F"][:, FEATURE_COLS.index(col)] for c in CONDS])
            r = np.concatenate([g[c][EST]["RL"] - g[c][EST]["FB"] for c in CONDS])
            fpr, tpr, _ = roc_curve(r > 0, s)
            tprs.append(np.interp(fpr_grid, fpr, tpr))
            aucs.append(float(auc(fpr, tpr)))
        roc[name] = {"fpr": fpr_grid.tolist(), "mean_tpr": np.mean(tprs, 0).tolist(),
                     "per_replicate_auroc": aucs, "auroc": float(np.mean(aucs))}
    out = {"label": STAMP, "source": [f"results/raw/val_{PRIMARY_IMPACT}.pkl",
                                      f"results/raw/test_{PRIMARY_IMPACT}.pkl"],
           "estimators": {"tail": "realised IS, statistic per replicate then averaged "
                                  "(as analysis.describe)",
                          "regret": "control-variate IS, RL - TWAP, all replicates pooled",
                          "roc": "harm = control-variate regret > 0; ROC per replicate, TPR "
                                 "averaged on a common FPR grid; AUROC = mean over replicates"},
           "tail_risk": tail, "regret": regret, "score_regret_hist2d": hist,
           "roc_harm": roc}
    p = paths.RESULTS / SUPP_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    return out


# --------------------------------------------------------------------------------------
# Figures from committed artifacts
# --------------------------------------------------------------------------------------


def s1_policy_selection(a: dict) -> Path:
    t = a["primary"]["table"]
    pols = ["RL", "ADAPT", "AC", "RCG", "ORACLE"]
    M = np.array([[t[p]["per_condition"][c] - t["FB"]["per_condition"][c] for c in CONDS]
                  for p in pols])
    lim = float(np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    best = np.argmin(M[:4], axis=0)  # among implementable policies (ORACLE is a bound)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.6 * lim else "#0b0b0b",
                    fontweight="bold" if (i < 4 and best[j] == i) else "normal")
    ax.set_xticks(range(len(CONDS)), CONDS, rotation=45, ha="right")
    ax.set_yticks(range(len(pols)), [LABEL.get(p, p) for p in pols])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="expected IS − TWAP (bp)")
    _stamp(ax, "S1. Policy cost relative to TWAP by regime (bold: lowest point estimate among "
               "implementable policies); test split, sqrt impact")
    return _save(fig, "S1_policy_selection.png")


def s2_gate_interventions(a: dict) -> Path:
    t = a["primary"]["table"]
    gates = ["RCG", "RCG_SEEN", "A9_RULE", "A_OOD", "A_ENS"]
    M = np.array([[t[g]["fallback_rate"][c] for c in CONDS] for g in gates])
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(CONDS)), CONDS, rotation=45, ha="right")
    ax.set_yticks(range(len(gates)),
                  [LABEL.get(g, g.replace("RCG_SEEN", "RCG seen-family (expl.)"))
                   for g in gates])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of orders sent to TWAP")
    _stamp(ax, "S2. Gate intervention frequency by condition (test split, sqrt impact)")
    return _save(fig, "S2_gate_interventions.png")


def s3_impact_sensitivity(a: dict) -> Path:
    ims = {PRIMARY_IMPACT: a["primary"], **a["impact_models"]}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(CONDS))
    for k, im in enumerate(IMPACT_MODELS):
        t = ims[im]["table"]
        d = [t["RL"]["per_condition"][c] - t["FB"]["per_condition"][c] for c in CONDS]
        ax.scatter(d, y + (k - 1) * 0.22, color=COLOR[im], s=22, zorder=3,
                   label=f"{im}{' (primary)' if im == PRIMARY_IMPACT else ''}")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, CONDS)
    ax.invert_yaxis()
    ax.set(xlabel="RL − TWAP expected IS (bp); > 0: RL worse than the fallback")
    ax.legend(title="impact model", fontsize=7.5, title_fontsize=7.5)
    _stamp(ax, "S3. RL's edge over TWAP per condition under each impact model "
               "(point estimates)")
    return _save(fig, "S3_impact_sensitivity.png")


# --------------------------------------------------------------------------------------
# Figures from the reduced per-episode data
# --------------------------------------------------------------------------------------


def s4_tail_risk(sd: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    x = np.arange(len(CONDS))
    pols = ("FB", "AC", "ADAPT", "RL")
    w = 0.2
    for k, p in enumerate(pols):
        v = [sd["tail_risk"][c][p]["cvar95"] for c in CONDS]
        ax.bar(x + (k - 1.5) * w, v, width=w * 0.92, color=COLOR[p], label=LABEL[p])
    ax.set_xticks(x, CONDS, rotation=45, ha="right")
    ax.set(ylabel="CVaR95 of realised IS (bp)")
    ax.legend(fontsize=7.5, ncols=4)
    _stamp(ax, "S4. Tail risk: mean cost of the worst 5% of orders, per condition")
    return _save(fig, "S4_tail_risk.png")


def s5_regret(sd: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, c in enumerate(CONDS):
        q = sd["regret"][c]["quantiles"]
        ax.plot([q["0.05"], q["0.95"]], [i, i], color=COLOR["RL"], lw=1.2, alpha=0.6)
        ax.plot([q["0.25"], q["0.75"]], [i, i], color=COLOR["RL"], lw=5, solid_capstyle="butt")
        ax.plot(q["0.5"], i, "|", color="white", ms=7, mew=1.6)
        ax.plot(sd["regret"][c]["mean"], i, "D", color=INK2, ms=4,
                label="mean" if i == 0 else None)
        ax.text(1.01, i, f"{sd['regret'][c]['share_positive']:.0%}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(CONDS)), CONDS)
    ax.invert_yaxis()
    ax.text(1.01, -0.9, "share RL worse", transform=ax.get_yaxis_transform(), fontsize=7,
            color=INK2)
    ax.set(xlabel="per-episode regret IS_RL − IS_TWAP (bp, control variate); "
                  "bar = IQR, line = 5-95%")
    ax.legend(fontsize=7.5, loc="lower left")
    _stamp(ax, "S5. Distribution of RL regret vs TWAP, all replicates and episodes")
    return _save(fig, "S5_rl_regret.png")


def s6_uncertainty_vs_regret(sd: dict, a: dict) -> Path:
    sc = a["primary"]["H3"]["scores"]
    names = {"RCG": "RCG", "OOD distance": "OOD distance"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (key, sname) in zip(axes, names.items(), strict=True):
        h = sd["score_regret_hist2d"][key]
        H = np.array(h["counts"], dtype=float).T
        Hm = np.ma.masked_where(H == 0, H)
        pc = ax.pcolormesh(h["x_edges"], h["y_edges"], Hm, cmap="Blues",
                           norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=H.max()))
        ax.axhline(0, color=INK2, lw=0.8)
        rho = sc[sname]["spearman"]
        auc = sc[sname]["auroc_harm"]
        ax.text(0.02, 0.97, f"Spearman ρ = {rho:+.3f}\nAUROC(regret > 0) = {auc:.3f}\n"
                            f"n = {h['n']:,} episode-replicates",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        ax.set(xlabel=f"{key} score at order arrival (full range)",
               ylabel="regret IS_RL − IS_TWAP (bp)")
        fig.colorbar(pc, ax=ax, label="episodes (log colour scale)")
        _stamp(ax, f"S6{'ab'[list(names).index(key)]}. {key} vs realised regret")
    fig.tight_layout()
    return _save(fig, "S6_uncertainty_vs_regret.png")


def s7_roc(sd: dict) -> Path:
    colors = {"RCG": COLOR["RCG"], "OOD distance": COLOR["A_OOD"],
              "Ensemble action std": COLOR["A_ENS"], "Shadow regret": COLOR["A_SHADOW"]}
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.9, ls="--", label="random ranking (0.500)")
    for name, r in sd["roc_harm"].items():
        lo, hi = min(r["per_replicate_auroc"]), max(r["per_replicate_auroc"])
        ax.plot(r["fpr"], r["mean_tpr"], color=colors[name], lw=1.6,
                label=f"{name}: {r['auroc']:.3f} (replicates {lo:.3f}–{hi:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel="false-positive rate (orders where RL beat TWAP, flagged)",
           ylabel="true-positive rate (orders where RL lost to TWAP, flagged)")
    n = sd["score_regret_hist2d"]["RCG"]["n"]
    ax.legend(title=f"AUROC, harm = regret > 0; n = {n:,}", fontsize=6.8, title_fontsize=7,
              loc="lower right")
    _stamp(ax, "S7. Harm-detection ROC, mean over 10 replicates")
    return _save(fig, "S7_harm_roc.png")


def make_supplementary_figures(recompute: bool = False) -> list[Path]:
    setup()
    a = json.loads((paths.RESULTS / "authoritative.json").read_text())
    made = [s1_policy_selection(a), s2_gate_interventions(a), s3_impact_sensitivity(a)]
    p = paths.RESULTS / SUPP_JSON
    if recompute or not p.exists() or "roc_harm" not in json.loads(p.read_text()):
        try:
            sd = compute_supplementary_data()
        except FileNotFoundError as e:
            print(f"S4-S7 skipped: {e}")
            return made
    else:
        sd = json.loads(p.read_text())
    made += [s4_tail_risk(sd), s5_regret(sd), s6_uncertainty_vs_regret(sd, a), s7_roc(sd)]
    return made


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild results/post_study/supplementary_data.json from results/raw")
    for f in make_supplementary_figures(ap.parse_args(argv).recompute):
        print("wrote", f.relative_to(paths.ROOT))
    return 0


__all__ = ["compute_supplementary_data", "main", "make_supplementary_figures"]


if __name__ == "__main__":
    raise SystemExit(main())
