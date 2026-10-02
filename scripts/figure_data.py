"""Descriptive per-condition summaries for figures -> results/figure_data.json.

Uses the frozen analysis functions on the confirmatory raw outputs; it computes no new
hypothesis tests (all inferential results live in results/authoritative.json).
"""

import json

import numpy as np

from adexec.analysis import EST, gated_results, stack
from adexec.config import MIXTURE_WEIGHTS, PRIMARY_IMPACT
from adexec.experiment import RAW, load
from adexec.features import FEATURE_COLS
from adexec.guard import ROOT
from adexec.stats import crossed_bootstrap, risk_coverage

POLS = ("RL", "FB", "AC", "ADAPT", "A_OOD", "A_ENS", "A9_RULE", "A3_BLEND", "A10_RAND", "RCG",
        "RCG_SEEN", "ORACLE")


def _mix_pooled(gres, p, key):
    """Pool episodes across conditions with mixture weights (replicate-averaged values)."""
    vals, wts = [], []
    for c, w in MIXTURE_WEIGHTS.items():
        x = stack(gres, c, p, key).mean(axis=0)
        vals.append(x)
        wts.append(np.full(x.size, w / x.size))
    return np.concatenate(vals), np.concatenate(wts)


def _wq(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(cw, q)])


def gate_eval(gres):
    """Descriptive mixture statistics per policy (realised cost distribution + CV mean)."""
    out = {}
    for p in POLS:
        x, w = _mix_pooled(gres, p, "IS")
        xc, _ = _mix_pooled(gres, p, EST)
        fb, _ = _mix_pooled(gres, "FB", EST)
        mu = float(np.sum(w * x))
        out[p] = {"mean_cv": float(np.sum(w * xc)), "mean_realised": mu,
                  "median_realised": _wq(x, w, 0.5),
                  "sd_realised": float(np.sqrt(np.sum(w * (x - mu) ** 2))),
                  "regret_vs_TWAP_cv": float(np.sum(w * (xc - fb)))}
        if p in ("RCG", "RCG_SEEN", "A9_RULE", "A_OOD", "A_ENS"):
            out[p]["fallback_rate_mixture"] = float(sum(
                MIXTURE_WEIGHTS[c] * np.mean([g[c]["fallback"][p].mean() for g in gres])
                for c in MIXTURE_WEIGHTS))
    return out


def harm_detection(gres):
    """Fallback decision as a detector of harmful episodes (CV regret > 0), pooled."""
    out = {}
    for g_name in ("RCG", "RCG_SEEN", "A9_RULE", "A_OOD", "A_ENS"):
        prec, rec, rate = [], [], []
        for g in gres:
            fb = np.concatenate([g[c]["fallback"][g_name] for c in MIXTURE_WEIGHTS])
            harm = np.concatenate([g[c][EST]["RL"] - g[c][EST]["FB"] > 0
                                   for c in MIXTURE_WEIGHTS])
            prec.append(harm[fb].mean() if fb.any() else np.nan)
            rec.append(fb[harm].mean())
            rate.append(harm.mean())
        out[g_name] = {"precision": float(np.nanmean(prec)), "recall": float(np.mean(rec)),
                       "harm_base_rate": float(np.mean(rate))}
    return out


def cv_audit(gres, B=1000, seed=11):
    """Unbiasedness and variance reduction of the control-variate estimator (descriptive)."""
    rng = np.random.default_rng(seed)
    out = {"bias_z": {}, "variance_ratio": {}}
    for c in MIXTURE_WEIGHTS:
        for p in ("RL", "FB", "ADAPT"):
            d = (stack(gres, c, p, "IS") - stack(gres, c, p, EST)).mean(axis=0)  # per episode
            out["bias_z"][f"{c}/{p}"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
        dr = stack(gres, c, "RL", "IS") - stack(gres, c, "FB", "IS")
        dc = stack(gres, c, "RL", EST) - stack(gres, c, "FB", EST)
        vr = dr.var(axis=1).mean() / dc.var(axis=1).mean()
        E = dr.shape[1]
        boots = []
        for _ in range(B):
            i = rng.integers(0, E, E)
            boots.append(dr[:, i].var(axis=1).mean() / dc[:, i].var(axis=1).mean())
        out["variance_ratio"][c] = [float(vr), float(np.quantile(boots, 0.025)),
                                    float(np.quantile(boots, 0.975))]
    z = np.array(list(out["bias_z"].values()))
    out["bias_summary"] = {"n_tests": int(z.size), "max_abs_z": float(np.abs(z).max()),
                           "n_abs_z_gt_1.96": int((np.abs(z) > 1.96).sum()),
                           "expected_false_alarms_at_5pct": float(0.05 * z.size)}
    return out


def main():
    val = load(RAW / f"val_{PRIMARY_IMPACT}.pkl")
    test = load(RAW / f"test_{PRIMARY_IMPACT}.pkl")
    gres = gated_results(val, test)
    out = {"per_condition": {}, "risk_coverage": {}, "fallback": {}}
    for c in MIXTURE_WEIGHTS:
        row = {}
        for a in ("RL", "ADAPT", "AC", "RCG", "RCG_SEEN", "A9_RULE", "A3_BLEND", "A_OOD"):
            r = crossed_bootstrap({c: stack(gres, c, a) - stack(gres, c, "FB")}, {c: 1.0},
                                  B=2000)
            row[a] = [r["estimate"], r["ci_low"], r["ci_high"]]
        row["TWAP_mean"] = float(stack(gres, c, "FB").mean())
        out["per_condition"][c] = row
        out["fallback"][c] = {g: float(np.mean([gr[c]["fallback"][g].mean() for gr in gres]))
                              for g in ("RCG", "RCG_SEEN", "A9_RULE", "A_OOD", "A_ENS")}
    conds = list(MIXTURE_WEIGHTS)
    regret = [np.concatenate([g[c][EST]["RL"] - g[c][EST]["FB"] for c in conds]) for g in gres]
    scores = {
        "RCG": lambda g: np.concatenate([g[c]["score_RCG"] for c in conds]),
        "RCG_SEEN": lambda g: np.concatenate([g[c]["score_RCG_SEEN"] for c in conds]),
        "A_OOD": lambda g: np.concatenate(
            [g[c]["F"][:, FEATURE_COLS.index("ood_maha")] for c in conds]),
        "A_ENS": lambda g: np.concatenate(
            [g[c]["F"][:, FEATURE_COLS.index("u_ens_s0")] for c in conds]),
    }
    for name, fn in scores.items():
        curves = [risk_coverage(fn(g), r, 41) for g, r in zip(gres, regret, strict=True)]
        out["risk_coverage"][name] = {
            "coverage": curves[0][0].tolist(),
            "total_regret": np.mean([c[2] for c in curves], 0).tolist()}
    # binned score vs realised (control-variate) regret, pooled over replicates
    bins = {}
    for name, fn in scores.items():
        s = np.concatenate([fn(g) for g in gres])
        r = np.concatenate(regret)
        edges = np.quantile(s, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, 9)
        bins[name] = {"score_rank_decile": list(range(1, 11)),
                      "mean_regret": [float(r[b == i].mean()) for i in range(10)],
                      "se": [float(r[b == i].std() / np.sqrt((b == i).sum())) for i in range(10)]}
    out["score_deciles"] = bins
    out["gate_eval"] = gate_eval(gres)
    out["harm_detection"] = harm_detection(gres)
    out["control_variate_audit"] = cv_audit(gres)
    (ROOT / "results" / "figure_data.json").write_text(json.dumps(out, indent=1))
    print("wrote results/figure_data.json")


if __name__ == "__main__":
    main()
