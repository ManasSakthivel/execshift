"""Selective-intervention gates and controls.

All gates decide once, at order arrival, between TRUST RL and FALL BACK (to TWAP).
Every tunable quantity (regressor, thresholds, blend weight) is fitted on the VALIDATION
split under leave-one-family-out (LOFO): the gate applied to test condition c of shift
family f was fitted on validation data that excludes family f entirely, so every shifted
test condition is an unseen *kind* of shift for the gate.  For the in-distribution test
condition the gate is fitted on all validation data.

Fitting targets use the control-variate IS (``IScv``; unbiased for expected IS, see
``sim.Execution.shortfall_cv_bps``).  Gates are applied to observable features only.

Gates
  RCG       proposed regret-calibrated gate: regression of counterfactual regret
            (IScv_RL - IScv_TWAP) on gate features; fall back iff predicted regret > 0
  A9_RULE   rule gate: fall back if any market summary is beyond its TRAIN-ID quantile q
  A_OOD     Mahalanobis OOD gate: fall back if OOD distance > TRAIN-ID quantile q
  A_ENS     ensemble-disagreement gate: fall back if action std > TRAIN-ID quantile q
  A_SHADOW  fall back iff shadow regret on the pre-trade window > 0 (no tuning)
Controls
  A3_BLEND  per-step blend w*RL + (1-w)*TWAP, w in {0.25, 0.5, 0.75} chosen on validation
  A10_RAND  random fallback with probability matched to RCG's fallback rate in each
            condition and replicate; reported as the exact expectation over coin flips
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONDITION_BY_NAME, MIXTURE_WEIGHTS
from .features import FEATURE_COLS

QGRID = (0.80, 0.90, 0.95, 0.99)
BLEND_W = (0.25, 0.5, 0.75)
RULE_COLS = {"m_rv": +1, "m_spread": +1, "m_depth": -1, "m_volume": -1, "m_size": +1}
COL = {c: i for i, c in enumerate(FEATURE_COLS)}
THRESHOLD_GATES = ("A9_RULE", "A_OOD", "A_ENS")
ALL_GATES = ("RCG", *THRESHOLD_GATES, "A_SHADOW")

# Candidate regressors for RCG; one is selected per replicate by nested LOFO on validation.
RCG_MODELS = {
    "hgb": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                                 min_samples_leaf=50, l2_regularization=1.0,
                                                 random_state=0),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
}


def family(cond: str) -> str:
    return CONDITION_BY_NAME[cond].family


def mixture_weights(conds) -> dict[str, float]:
    """Pre-registered mixture restricted to `conds` (ID 0.5, shifts share 0.5 equally)."""
    shifts = [c for c in conds if c != "ID"]
    w = {c: 0.5 / len(shifts) for c in shifts} if shifts else {}
    if "ID" in conds:
        w["ID"] = 0.5 if shifts else 1.0
    return w


def mixture_mean(values: dict[str, np.ndarray], conds) -> float:
    w = mixture_weights(conds)
    return float(sum(w[c] * values[c].mean() for c in w))


def training_conds(val_conds, held_out_family: str | None):
    return [c for c in val_conds if held_out_family is None or family(c) != held_out_family
            or c == "ID"]


def regret(d: dict) -> np.ndarray:
    return d["IScv"]["RL"] - d["IScv"]["FB"]


def fit_regressor(val: dict, conds, kind: str):
    X = np.concatenate([val[c]["F"] for c in conds])
    y = np.concatenate([regret(val[c]) for c in conds])
    mw = mixture_weights(conds)
    sw = np.concatenate([np.full(len(val[c]["F"]), mw[c] / len(val[c]["F"])) for c in conds])
    reg = RCG_MODELS[kind]()
    if kind == "ridge":
        reg.fit(X, y, ridge__sample_weight=sw * len(sw))
    else:
        reg.fit(X, y, sample_weight=sw * len(sw))
    return reg


def select_rcg_model(val: dict) -> dict:
    """Nested LOFO on validation only: for every shift family f, fit on the other families
    (+ ID) and gate family f; ID is scored by 2-fold cross-fitting. Pick the model with the
    lowest mixture mean IScv."""
    conds = list(val)
    fams = sorted({family(c) for c in conds} - {"ID"})
    scores = {}
    for kind in RCG_MODELS:
        gated = {}
        for f in fams:
            reg = fit_regressor(val, training_conds(conds, f), kind)
            for c in conds:
                if family(c) == f:
                    fb = reg.predict(val[c]["F"]) > 0
                    gated[c] = np.where(fb, val[c]["IScv"]["FB"], val[c]["IScv"]["RL"])
        n = len(val["ID"]["F"])
        halves = [np.arange(n) % 2 == 0, np.arange(n) % 2 == 1]
        g_id = np.empty(n)
        for h_fit, h_eval in (halves, halves[::-1]):
            sub = {c: d for c, d in val.items() if c != "ID"}
            sub["ID"] = {"F": val["ID"]["F"][h_fit],
                         "IScv": {k: v[h_fit] for k, v in val["ID"]["IScv"].items()}}
            reg = fit_regressor(sub, conds, kind)
            fb = reg.predict(val["ID"]["F"][h_eval]) > 0
            g_id[h_eval] = np.where(fb, val["ID"]["IScv"]["FB"][h_eval],
                                    val["ID"]["IScv"]["RL"][h_eval])
        gated["ID"] = g_id
        scores[kind] = mixture_mean(gated, conds)
    return {"selected": min(scores, key=scores.get), "nested_lofo_mixture_IScv": scores}


class GateSuite:
    """Fit on one replicate's validation data; apply to its test data."""

    def __init__(self, train_feats: np.ndarray, rcg_kind: str = "hgb"):
        self.ref = train_feats  # TRAIN-ID reference distributions for quantile thresholds
        self.rcg_kind = rcg_kind

    def _quantile_rule(self, name: str, q: float):
        ref = self.ref
        if name == "A_OOD":
            thr = np.quantile(ref[:, COL["ood_maha"]], q)
            return lambda F: F[:, COL["ood_maha"]] > thr
        if name == "A_ENS":
            thr = np.quantile(ref[:, COL["u_ens_s0"]], q)
            return lambda F: F[:, COL["u_ens_s0"]] > thr
        assert name == "A9_RULE"
        lims = {c: np.quantile(ref[:, COL[c]], q if s > 0 else 1 - q)
                for c, s in RULE_COLS.items()}

        def rule(F):
            out = np.zeros(F.shape[0], dtype=bool)
            for c, s in RULE_COLS.items():
                out |= (F[:, COL[c]] > lims[c]) if s > 0 else (F[:, COL[c]] < lims[c])
            return out
        return rule

    def fit(self, val: dict, held_out_family: str | None) -> dict:
        conds = training_conds(list(val), held_out_family)
        fitted = {}
        reg = fit_regressor(val, conds, self.rcg_kind)
        fitted["RCG"] = {"score": reg.predict, "decide": lambda F, r=reg: r.predict(F) > 0}
        for name in THRESHOLD_GATES:
            best = None
            for q in QGRID:
                rule = self._quantile_rule(name, q)
                gated = {c: np.where(rule(val[c]["F"]), val[c]["IScv"]["FB"],
                                     val[c]["IScv"]["RL"]) for c in conds}
                s = mixture_mean(gated, conds)
                if best is None or s < best[0]:
                    best = (s, q, rule)
            fitted[name] = {"decide": best[2], "q": best[1]}
        fitted["A_SHADOW"] = {"decide": lambda F: F[:, COL["shadow_regret"]] > 0}
        blend = {w: mixture_mean({c: val[c]["IScv"][f"BLEND_{w}"] for c in conds}, conds)
                 for w in BLEND_W}
        fitted["A3_BLEND"] = {"w": min(blend, key=blend.get)}
        return fitted


def apply_gates(suite: GateSuite, val: dict, test: dict, lofo: bool = True) -> dict:
    """Per test condition: gated IS (realised and control-variate), fallback decisions, RCG
    scores and fitted choices.  ``lofo=False`` gives the exploratory *seen-family* variant
    in which the gate was fitted on validation data of every family, including the test
    condition's own family (different episodes)."""
    if lofo:
        fits = {fam: suite.fit(val, fam) for fam in {family(c) for c in test} - {"ID"}}
        fits["ID"] = suite.fit(val, None)
    else:
        f_all = suite.fit(val, None)
        fits = {family(c): f_all for c in test}
    out = {}
    for c, d in test.items():
        fit = fits[family(c)]
        F = d["F"]
        res = {"IS": {}, "IScv": {}, "fallback": {}, "meta": {}}
        for g in ALL_GATES:
            fb = fit[g]["decide"](F)
            res["fallback"][g] = fb
            if "q" in fit[g]:
                res["meta"][g + "_q"] = fit[g]["q"]
        res["score_RCG"] = fit["RCG"]["score"](F)
        p = float(res["fallback"]["RCG"].mean())
        w = fit["A3_BLEND"]["w"]
        res["meta"].update({"A10_p": p, "A3_w": w})
        for key in ("IS", "IScv"):
            IS = d[key]
            for g in ALL_GATES:
                res[key][g] = np.where(res["fallback"][g], IS["FB"], IS["RL"])
            res[key]["A10_RAND"] = (1 - p) * IS["RL"] + p * IS["FB"]  # exact expectation
            res[key]["A3_BLEND"] = IS[f"BLEND_{w}"]
            res[key]["ORACLE"] = np.where(d["IScv"]["RL"] <= d["IScv"]["FB"], IS["RL"], IS["FB"])
        out[c] = res
    return out


__all__ = ["GateSuite", "apply_gates", "select_rcg_model", "mixture_weights",
           "MIXTURE_WEIGHTS", "ALL_GATES"]
