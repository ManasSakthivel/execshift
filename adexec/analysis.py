"""Pre-registered analysis (H1-H6).  Pure functions of the raw evaluation pickles.

``analyze(fit_raw, eval_raw)`` fits all gates on ``fit_raw`` (validation) and evaluates on
``eval_raw`` (test in the confirmatory run; a held-out half of validation during
development, see ``split_raw``).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .config import ALPHA, CONDITION_BY_NAME, MIXTURE_WEIGHTS
from .features import FEATURE_COLS
from .gates import ALL_GATES, GateSuite, apply_gates, select_rcg_model
from .stats import (
    auprc,
    aurc,
    auroc,
    crossed_bootstrap,
    holm,
    risk_coverage,
    seed_level_wilcoxon,
)

SESOI = 0.25  # bp, pre-registered
EST = "IScv"  # primary estimator of expected IS (control variate); "IS" = realised sensitivity
H5_CONTROLS = ("A3_BLEND", "A9_RULE", "A10_RAND", "FB")
GATE_POLICIES = ("RL", "FB", "ADAPT", "AC", "RL1", "RCG", "A3_BLEND", "A9_RULE", "A10_RAND",
                 "A_OOD", "A_ENS", "A_SHADOW", "RCG_SEEN", "ORACLE")


def split_raw(raw: dict, part: int) -> dict:
    """Development only: halve every validation condition (even / odd episodes)."""
    out = {**raw, "results": []}
    for rep in raw["results"]:
        conds = {}
        for c, d in rep["conds"].items():
            sl = slice(part, None, 2)
            conds[c] = {"IS": {k: v[sl] for k, v in d["IS"].items()},
                        "IScv": {k: v[sl] for k, v in d["IScv"].items()}, "F": d["F"][sl],
                        "rl_trade_sd": d["rl_trade_sd"][sl]}
        out["results"].append({"train_F": rep["train_F"], "conds": conds})
    return out


def gated_results(fit_raw: dict, eval_raw: dict) -> list[dict]:
    """Per replicate: {cond: {"IS": {policy: array}, "fallback": {...}, "score_RCG": ...}}."""
    out = []
    for rf, re in zip(fit_raw["results"], eval_raw["results"], strict=True):
        sel = select_rcg_model(rf["conds"])
        suite = GateSuite(rf["train_F"], rcg_kind=sel["selected"])
        g = apply_gates(suite, rf["conds"], re["conds"])
        seen = apply_gates(suite, rf["conds"], re["conds"], lofo=False)
        for c, d in re["conds"].items():
            for key in ("IS", "IScv"):
                g[c][key]["RCG_SEEN"] = seen[c][key]["RCG"]
            g[c]["fallback"]["RCG_SEEN"] = seen[c]["fallback"]["RCG"]
            g[c]["score_RCG_SEEN"] = seen[c]["score_RCG"]
            for key in ("IS", "IScv"):
                for k, v in d[key].items():
                    g[c][key].setdefault(k, v)
            g[c]["F"] = d["F"]
            g[c]["meta"]["rcg_model"] = sel["selected"]
            g[c]["meta"]["rcg_selection"] = sel["nested_lofo_mixture_IScv"]
        out.append(g)
    return out


def stack(gres: list[dict], cond: str, policy: str, key: str = EST) -> np.ndarray:
    return np.stack([g[cond][key][policy] for g in gres])  # (R, E)


def describe_policy(gres, cond: str, policy: str) -> dict:
    d = describe(stack(gres, cond, policy, "IS"))  # realised-cost distribution (risk metrics)
    d["mean_cv"] = float(stack(gres, cond, policy).mean())
    d["per_replicate_mean_cv"] = stack(gres, cond, policy).mean(axis=1).round(4).tolist()
    return d


def describe(x: np.ndarray) -> dict:
    """x: (R, E) IS values. Mean of per-replicate statistics, plus per-replicate means."""
    per_rep = x.mean(axis=1)
    q = np.quantile(x, 0.95, axis=1, keepdims=True)
    cvar = np.array([row[row >= qq].mean() for row, qq in zip(x, q[:, 0], strict=True)])
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "sd": float(x.std(axis=1).mean()), "cvar95": float(cvar.mean()),
            "per_replicate_mean": per_rep.round(4).tolist(),
            "replicate_sd": float(per_rep.std(ddof=1))}


def compare(gres, a: str, b: str, weights: dict, sensitivity: bool = True) -> dict:
    diffs = {c: stack(gres, c, a) - stack(gres, c, b) for c in weights}
    res = crossed_bootstrap(diffs, weights)
    per_rep = sum(weights[c] * diffs[c].mean(axis=1) for c in weights) / sum(weights.values())
    res["per_replicate"] = per_rep.round(4).tolist()
    res["wilcoxon_replicates"] = seed_level_wilcoxon(per_rep)
    if sensitivity:  # same comparison with the realised-IS estimator
        dr = {c: stack(gres, c, a, "IS") - stack(gres, c, b, "IS") for c in weights}
        r = crossed_bootstrap(dr, weights, B=2000)
        res["realised"] = {k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_value")}
    return res


def _support(res: dict, direction: int = -1) -> bool:
    return bool(res["reject"] and direction * res["estimate"] > SESOI)


# --------------------------------------------------------------------------------------
def h1(gres) -> dict:
    w = {"ID": 1.0}
    comps = {"RL-TWAP": compare(gres, "RL", "FB", w), "RL-AC": compare(gres, "RL", "AC", w),
             "RL-ADAPT": compare(gres, "RL", "ADAPT", w)}
    hm = holm({k: v["p_value"] for k, v in comps.items()}, ALPHA)
    for k in comps:
        comps[k].update(hm[k])
    extra = {"RL1-TWAP": compare(gres, "RL1", "FB", w),
             "ADAPT-TWAP": compare(gres, "ADAPT", "FB", w),
             "AC-TWAP": compare(gres, "AC", "FB", w)}
    desc = {p: describe_policy(gres, "ID", p) for p in ("RL", "RL1", "FB", "AC", "ADAPT")}
    supported = _support(comps["RL-TWAP"]) and _support(comps["RL-AC"])
    return {"comparisons": comps, "secondary": extra, "describe": desc,
            "supported": supported}


def h2(gres) -> dict:
    out = {}
    for c in MIXTURE_WEIGHTS:
        if c == "ID":
            continue
        diffs = {c: stack(gres, c, "RL") - stack(gres, c, "FB"),
                 "ID": stack(gres, "ID", "RL") - stack(gres, "ID", "FB")}
        res = crossed_bootstrap(diffs, {c: 1.0, "ID": -1.0}, normalize=False)
        res.pop("d_z", None)
        res["regret_in_condition"] = float(diffs[c].mean())
        res["regret_ID"] = float(diffs["ID"].mean())
        out[c] = res
    hm = holm({c: v["p_value"] for c, v in out.items()}, ALPHA)
    for c in out:
        out[c].update(hm[c])
    supported = any(v["reject"] and v["estimate"] > SESOI for v in out.values())
    harmful = [c for c, v in out.items() if v["regret_in_condition"] > 0]
    return {"conditions": out, "supported": supported, "conditions_where_RL_worse_than_TWAP":
            harmful}


SCORES = {"RCG": None, "RCG (seen-family, exploratory)": "SEEN", "OOD distance": "ood_maha",
          "Ensemble action std": "u_ens_s0",
          "Ensemble value std": "u_val_std", "Shadow regret": "shadow_regret"}


def _h3_metrics(score: np.ndarray, regret: np.ndarray) -> dict:
    top = regret > np.quantile(regret, 0.9)
    pos = regret > 0
    return {"spearman": stats.spearmanr(score, regret)[0], "auroc_harm": auroc(score, pos),
            "auroc_top10": auroc(score, top), "auprc_harm": auprc(score, pos),
            "auprc_harm_base": float(pos.mean()), "aurc": aurc(score, regret),
            "aurc_random": float(regret.sum() / regret.size * 0.5)}


def h3(gres, B: int = 1000, seed: int = 7) -> dict:
    conds = list(MIXTURE_WEIGHTS)
    R = len(gres)

    def scores_of(g, name):
        if SCORES[name] is None:
            return np.concatenate([g[c]["score_RCG"] for c in conds])
        if SCORES[name] == "SEEN":
            return np.concatenate([g[c]["score_RCG_SEEN"] for c in conds])
        j = FEATURE_COLS.index(SCORES[name])
        return np.concatenate([g[c]["F"][:, j] for c in conds])

    regret = [np.concatenate([g[c][EST]["RL"] - g[c][EST]["FB"] for c in conds]) for g in gres]
    regret_real = [np.concatenate([g[c]["IS"]["RL"] - g[c]["IS"]["FB"] for c in conds])
                   for g in gres]
    sizes = [gres[0][c]["IS"]["RL"].size for c in conds]
    offs = np.cumsum([0, *sizes])
    out = {}
    rng = np.random.default_rng(seed)
    boot_idx = []
    for _ in range(B):
        reps = rng.integers(0, R, R)
        eps = np.concatenate([o + rng.integers(0, n, n) for o, n in zip(offs[:-1], sizes,
                                                                        strict=True)])
        boot_idx.append((reps, eps))
    for name in SCORES:
        sc = [scores_of(g, name) for g in gres]
        per_rep = [_h3_metrics(s, r) for s, r in zip(sc, regret, strict=True)]
        point = {k: float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}
        bs = {k: [] for k in ("spearman", "auroc_harm", "auroc_top10")}
        if name == "RCG" or B <= 200:
            nb = B
        else:
            nb = 200  # exploratory scores: smaller bootstrap
        for reps, eps in boot_idx[:nb]:
            ms = [_h3_core(sc[r][eps], regret[r][eps]) for r in reps]
            for k in bs:
                bs[k].append(np.mean([m[k] for m in ms]))
        point["ci"] = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
                       for k, v in bs.items()}
        point["per_replicate_spearman"] = [round(float(m["spearman"]), 4) for m in per_rep]
        real = [_h3_core(s, r) for s, r in zip(sc, regret_real, strict=True)]
        point["realised_regret"] = {k: float(np.mean([m[k] for m in real])) for k in real[0]}
        out[name] = point
    r0 = out["RCG"]
    supported = bool(r0["ci"]["spearman"][0] > 0 and r0["ci"]["auroc_harm"][0] > 0.5)
    # within-condition discrimination (does the score rank episodes *inside* a condition?)
    within = {}
    for c in conds:
        rho = [stats.spearmanr(g[c]["score_RCG"], g[c][EST]["RL"] - g[c][EST]["FB"])[0]
               for g in gres]
        within[c] = float(np.mean(rho))
    # risk-coverage and calibration curves (replicate-averaged)
    rc = [risk_coverage(scores_of(g, "RCG"), r) for g, r in zip(gres, regret, strict=True)]
    cal = calibration(gres)
    return {"scores": out, "supported": supported, "within_condition_spearman": within,
            "risk_coverage": {"coverage": rc[0][0].tolist(),
                              "selective_mean_regret": np.mean([x[1] for x in rc], 0).tolist(),
                              "total_regret_per_episode": np.mean([x[2] for x in rc], 0).tolist()},
            "calibration": cal}


def _h3_core(score, regret):
    top = regret > np.quantile(regret, 0.9)
    return {"spearman": stats.spearmanr(score, regret)[0], "auroc_harm": auroc(score, regret > 0),
            "auroc_top10": auroc(score, top)}


def calibration(gres, n_bins: int = 10) -> dict:
    pred, real = [], []
    for g in gres:
        s = np.concatenate([g[c]["score_RCG"] for c in MIXTURE_WEIGHTS])
        r = np.concatenate([g[c][EST]["RL"] - g[c][EST]["FB"] for c in MIXTURE_WEIGHTS])
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
        pred.append([s[b == i].mean() for i in range(n_bins)])
        real.append([r[b == i].mean() for i in range(n_bins)])
    return {"predicted": np.mean(pred, 0).tolist(), "realised": np.mean(real, 0).tolist()}


def h45(gres) -> dict:
    w = MIXTURE_WEIGHTS
    h4 = compare(gres, "RCG", "RL", w)
    h4.update(holm({"RCG-RL": h4["p_value"]}, ALPHA)["RCG-RL"])
    h5 = {f"RCG-{c}": compare(gres, "RCG", c, w) for c in H5_CONTROLS}
    hm = holm({k: v["p_value"] for k, v in h5.items()}, ALPHA)
    for k in h5:
        h5[k].update(hm[k])
    expl = {f"RCG-{c}": compare(gres, "RCG", c, w) for c in ("A_OOD", "A_ENS", "A_SHADOW",
                                                              "ADAPT", "ORACLE")}
    expl["RCG_SEEN-RL"] = compare(gres, "RCG_SEEN", "RL", w)
    expl["RCG_SEEN-RCG"] = compare(gres, "RCG_SEEN", "RCG", w)
    h5_supported = all(v["reject"] and v["estimate"] < 0 for v in h5.values())
    return {"H4": {"comparison": h4, "supported": _support(h4)},
            "H5": {"comparisons": h5, "supported": h5_supported,
                   "beats": [k for k, v in h5.items() if v["reject"] and v["estimate"] < 0]},
            "exploratory": expl}


def mixture_table(gres) -> dict:
    """Mixture mean IS (+ per-condition means) and fallback rates for every policy / gate."""
    tab = {}
    for p in GATE_POLICIES:
        per_c = {c: float(stack(gres, c, p).mean()) for c in MIXTURE_WEIGHTS}
        mix = sum(MIXTURE_WEIGHTS[c] * per_c[c] for c in MIXTURE_WEIGHTS)
        per_rep = sum(MIXTURE_WEIGHTS[c] * stack(gres, c, p).mean(axis=1) for c in MIXTURE_WEIGHTS)
        tab[p] = {"mixture_mean": float(mix), "per_condition": per_c,
                  "per_replicate_mixture": per_rep.round(4).tolist()}
        if p in (*ALL_GATES, "RCG_SEEN"):
            tab[p]["fallback_rate"] = {c: float(np.mean([g[c]["fallback"][p].mean()
                                                         for g in gres]))
                                       for c in MIXTURE_WEIGHTS}
    tab["meta"] = {c: {k: [g[c]["meta"].get(k) for g in gres] for k in gres[0][c]["meta"]}
                   for c in MIXTURE_WEIGHTS}
    tab["realised_mixture_mean"] = {
        p: float(sum(MIXTURE_WEIGHTS[c] * stack(gres, c, p, "IS").mean() for c in MIXTURE_WEIGHTS))
        for p in GATE_POLICIES}
    return tab


def per_family_gate(gres) -> dict:
    """RCG vs controls within each family (robustness across market-dynamics axes)."""
    fams = {}
    for c in MIXTURE_WEIGHTS:
        fams.setdefault(CONDITION_BY_NAME[c].family, []).append(c)
    out = {}
    for f, cs in fams.items():
        w = {c: 1.0 for c in cs}
        out[f] = {}
        for k in ("RL", *H5_CONTROLS):
            r = compare(gres, "RCG", k, w, sensitivity=False)
            out[f][k] = {q: r[q] for q in ("estimate", "ci_low", "ci_high", "p_value")}
    return out


def analyze(fit_raw: dict, eval_raw: dict, full: bool = True) -> dict:
    gres = gated_results(fit_raw, eval_raw)
    res = {"impact_model": eval_raw["impact_model"], "latency_s": eval_raw.get("latency_s"),
           "n_replicates": len(gres), "replicates": eval_raw["replicates"],
           "n_episodes": {c: int(gres[0][c]["IS"]["RL"].size) for c in MIXTURE_WEIGHTS},
           "sesoi_bp": SESOI, "H45": h45(gres), "table": mixture_table(gres)}
    if full:
        res.update({"H1": h1(gres), "H2": h2(gres), "H3": h3(gres),
                    "per_family": per_family_gate(gres)})
    return res
