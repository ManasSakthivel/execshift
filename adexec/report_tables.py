"""Markdown tables generated from results/authoritative.json (no hand-entered numbers)."""

from __future__ import annotations

from .config import MIXTURE_WEIGHTS

POL = {"RL": "RL (5-member ensemble)", "RL1": "RL (single member)", "FB": "TWAP",
       "AC": "AC (tuned)", "ADAPT": "ADAPT (tuned)", "RCG": "RCG (proposed gate)",
       "A3_BLEND": "A3 blend", "A9_RULE": "A9 rule gate", "A10_RAND": "A10 random (matched)",
       "A_OOD": "OOD gate", "A_ENS": "Ensemble-disagreement gate", "A_SHADOW": "Shadow gate",
       "RCG_SEEN": "RCG, seen-family (exploratory)", "ORACLE": "Oracle (not achievable)"}


def f(x, d=2):
    return f"{x:+.{d}f}" if x is not None else "—"


def ci(r, d=2):
    return f"{r['estimate']:+.{d}f} [{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]"


def p(x):
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"


def yn(b):
    return "**yes**" if b else "no"


def t_id_describe(a):
    d = a["primary"]["H1"]["describe"]
    rows = ["| Policy | Mean IS, primary (CV) | Mean IS, realised | Median | SD | CVaR95 | "
            "SD across replicates |", "|---|---|---|---|---|---|---|"]
    for k in ("RL", "RL1", "ADAPT", "AC", "FB"):
        x = d[k]
        rows.append(f"| {POL[k]} | {x['mean_cv']:.3f} | {x['mean']:.3f} | {x['median']:.3f} | "
                    f"{x['sd']:.2f} | {x['cvar95']:.2f} | {x['replicate_sd']:.3f} |")
    return rows


def t_h1(a):
    h = a["primary"]["H1"]
    rows = ["| Comparison (ID test) | Δ expected IS, bp [95% CI] | p (Holm) | d_z | "
            "Realised-IS Δ [95% CI] | Replicates favouring RL | Wilcoxon p (replicates) |",
            "|---|---|---|---|---|---|---|"]
    for k, r in h["comparisons"].items():
        w = r["wilcoxon_replicates"]
        rows.append(f"| {k.replace('FB', 'TWAP')} | {ci(r)} | {p(r['p_holm'])} | "
                    f"{r['d_z']:+.3f} | {ci(r['realised'])} | {w['n_negative']}/{w['n']} | "
                    f"{p(w['p_value'])} |")
    for k, r in h["secondary"].items():
        rows.append(f"| {k} (secondary) | {ci(r)} | {p(r['p_value'])} (raw) | "
                    f"{r['d_z']:+.3f} | {ci(r['realised'])} | — | — |")
    return rows


def t_seeds(a):
    h = a["primary"]["H1"]["comparisons"]
    keys = list(h)
    rows = ["| Replicate | " + " | ".join(k.replace("FB", "TWAP") for k in keys) + " |",
            "|---|" + "---|" * len(keys)]
    for i in range(a["primary"]["n_replicates"]):
        rows.append(f"| {i} | " + " | ".join(f"{h[k]['per_replicate'][i]:+.3f}" for k in keys)
                    + " |")
    return rows


def t_h2(a):
    h = a["primary"]["H2"]["conditions"]
    rows = ["| Condition | RL − TWAP in condition (bp) | RL − TWAP in ID (bp) | Degradation "
            "D_c [95% CI] | p (Holm, 12) | D_c > SESOI and significant |",
            "|---|---|---|---|---|---|"]
    for c, r in h.items():
        rows.append(f"| {c} | {r['regret_in_condition']:+.2f} | {r['regret_ID']:+.2f} | "
                    f"{ci(r)} | {p(r['p_holm'])} | {yn(r['reject'] and r['estimate'] > 0.25)} |")
    return rows


def t_h3(a):
    s = a["primary"]["H3"]["scores"]
    rows = ["| Score | Spearman ρ [95% CI] | AUROC regret>0 [95% CI] | AUROC top-10% "
            "[95% CI] | AUPRC (base rate) | ρ vs realised regret |", "|---|---|---|---|---|---|"]
    for n, r in s.items():
        c = r["ci"]
        rows.append(f"| {n} | {r['spearman']:+.3f} [{c['spearman'][0]:+.3f}, "
                    f"{c['spearman'][1]:+.3f}] | {r['auroc_harm']:.3f} [{c['auroc_harm'][0]:.3f}, "
                    f"{c['auroc_harm'][1]:.3f}] | {r['auroc_top10']:.3f} "
                    f"[{c['auroc_top10'][0]:.3f}, {c['auroc_top10'][1]:.3f}] | "
                    f"{r['auprc_harm']:.3f} ({r['auprc_harm_base']:.3f}) | "
                    f"{r['realised_regret']['spearman']:+.3f} |")
    return rows


def t_within(a):
    w = a["primary"]["H3"]["within_condition_spearman"]
    return ["| " + " | ".join(w) + " |", "|" + "---|" * len(w),
            "| " + " | ".join(f"{v:+.3f}" for v in w.values()) + " |"]


def t_gates(res):
    h = res["H45"]
    rows = ["| Comparison (deployment mixture) | Δ expected IS, bp [95% CI] | p (Holm) | "
            "Realised-IS Δ [95% CI] | Significant & < 0 |", "|---|---|---|---|---|"]
    r = h["H4"]["comparison"]
    rows.append(f"| H4: RCG − always-RL | {ci(r, 3)} | {p(r['p_holm'])} | "
                f"{ci(r['realised'], 3)} | {yn(r['reject'] and r['estimate'] < 0)} |")
    for k, r in h["H5"]["comparisons"].items():
        name = POL.get(k.split("-", 1)[1], k)
        rows.append(f"| H5: RCG − {name} | {ci(r, 3)} | {p(r['p_holm'])} | "
                    f"{ci(r['realised'], 3)} | {yn(r['reject'] and r['estimate'] < 0)} |")
    for k, r in h["exploratory"].items():
        a_, b_ = k.split("-", 1)
        rows.append(f"| exploratory: {POL.get(a_, a_)} − {POL.get(b_, b_)} | {ci(r, 3)} | "
                    f"{p(r['p_value'])} (raw) | {ci(r['realised'], 3)} | — |")
    return rows


def t_mixture(res):
    t = res["table"]
    conds = list(MIXTURE_WEIGHTS)
    rows = ["| Policy | Mixture IS (CV) | Mixture IS (realised) | " + " | ".join(conds) + " |",
            "|---|---|---|" + "---|" * len(conds)]
    for k in POL:
        if k not in t:
            continue
        rows.append(f"| {POL[k]} | {t[k]['mixture_mean']:.3f} | "
                    f"{t['realised_mixture_mean'][k]:.3f} | "
                    + " | ".join(f"{t[k]['per_condition'][c]:.2f}" for c in conds) + " |")
    return rows


def t_fallback(res):
    t = res["table"]
    conds = list(MIXTURE_WEIGHTS)
    rows = ["| Gate | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for k in ("RCG", "RCG_SEEN", "A9_RULE", "A_OOD", "A_ENS", "A_SHADOW"):
        rows.append(f"| {POL[k]} | " + " | ".join(f"{t[k]['fallback_rate'][c]:.1%}"
                                                  for c in conds) + " |")
    meta = t["meta"]
    rows += ["", "Fitted choices on validation, as the range over replicates and LOFO folds: "
             + "; ".join(sorted({f"{c}: A3 w ∈ {sorted(set(meta[c]['A3_w']))}, A9 q ∈ "
                                 f"{sorted(set(meta[c].get('A9_RULE_q', [])))}"
                                 for c in ("ID", "LIQ-sev", "ALPHA-sev")}))
             + f"; RCG model class per replicate: {meta['ID']['rcg_model']}."]
    return rows


def t_family(a):
    pf = a["primary"]["per_family"]
    keys = ("RL", "A3_BLEND", "A9_RULE", "A10_RAND", "FB")
    rows = ["| Family | " + " | ".join(f"RCG − {POL[k]}" for k in keys) + " |",
            "|---|" + "---|" * len(keys)]
    for fam, r in pf.items():
        rows.append(f"| {fam} | " + " | ".join(ci(r[k]) for k in keys) + " |")
    return rows


def t_robust(a):
    rows = ["| Setting | RL − TWAP (ID) | H4: RCG − RL | RCG − A3 | RCG − A9 | RCG − A10 | "
            "RCG − TWAP | H3 ρ (RCG) |", "|---|---|---|---|---|---|---|---|"]
    sets = [("impact = sqrt (primary)", a["primary"])]
    sets += [(f"impact = {k}", v) for k, v in a["impact_models"].items()]
    sets += [(f"latency = {k} s (sqrt)", v) for k, v in a["latency"].items()]
    for name, r in sets:
        h5 = r["H45"]["H5"]["comparisons"]
        h1 = ci(r["H1"]["comparisons"]["RL-TWAP"]) if "H1" in r else "—"
        h3 = f"{r['H3']['scores']['RCG']['spearman']:+.3f}" if "H3" in r else "—"
        rows.append(f"| {name} | {h1} | {ci(r['H45']['H4']['comparison'], 3)} | "
                    f"{ci(h5['RCG-A3_BLEND'], 3)} | {ci(h5['RCG-A9_RULE'], 3)} | "
                    f"{ci(h5['RCG-A10_RAND'], 3)} | {ci(h5['RCG-FB'], 3)} | {h3} |")
    return rows


def t_hypotheses(a):
    pr = a["primary"]
    h2 = pr["H2"]
    rows = ["| Hypothesis | Pre-registered criterion | Result | Verdict |", "|---|---|---|---|",
            f"| H1 in-distribution | RL − TWAP and RL − AC < −0.25 bp, Holm-significant | "
            f"RL − TWAP {ci(pr['H1']['comparisons']['RL-TWAP'])}; RL − AC "
            f"{ci(pr['H1']['comparisons']['RL-AC'])} | "
            f"{'SUPPORTED' if pr['H1']['supported'] else 'NOT SUPPORTED'} |",
            f"| H2 degradation | ≥ 1 condition with D_c > 0.25 bp, Holm-significant | "
            f"{sum(v['reject'] and v['estimate'] > 0.25 for v in h2['conditions'].values())}/12 "
            f"conditions; RL worse than TWAP in: "
            f"{', '.join(h2['conditions_where_RL_worse_than_TWAP']) or 'none'} | "
            f"{'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'} |",
            f"| H3 score predicts regret | ρ CI > 0 and AUROC CI > 0.5 | ρ = "
            f"{pr['H3']['scores']['RCG']['spearman']:+.3f} "
            f"[{pr['H3']['scores']['RCG']['ci']['spearman'][0]:+.3f}, "
            f"{pr['H3']['scores']['RCG']['ci']['spearman'][1]:+.3f}], AUROC "
            f"{pr['H3']['scores']['RCG']['auroc_harm']:.3f} | "
            f"{'SUPPORTED' if pr['H3']['supported'] else 'NOT SUPPORTED'} |",
            f"| H4 gate beats always-RL | RCG − RL < −0.25 bp, significant | "
            f"{ci(pr['H45']['H4']['comparison'], 3)} | "
            f"{'SUPPORTED' if pr['H45']['H4']['supported'] else 'NOT SUPPORTED'} |",
            f"| H5 gate beats A3, A9, A10, TWAP | all four Holm-significant and < 0 | beats: "
            f"{', '.join(k.split('-', 1)[1] for k in pr['H45']['H5']['beats']) or 'none'} | "
            f"{'SUPPORTED' if pr['H45']['H5']['supported'] else 'NOT SUPPORTED (partial)' if pr['H45']['H5']['beats'] else 'NOT SUPPORTED'} |"]
    return rows


def render_tables(a: dict) -> str:
    out = ["# Result tables", "",
           "*Generated by `scripts/build_report.py` from `results/authoritative.json`. Do not "
           "edit by hand.* All data are synthetic. IS is in bp of arrival notional; lower is "
           "better; Δ < 0 favours the first-named policy. Primary estimator: control-variate "
           "IS (CV), with realised IS as sensitivity. CIs are 95% crossed (replicate × "
           "episode) bootstraps.", ""]
    sections = [
        ("T1. Pre-registered hypotheses", t_hypotheses(a)),
        ("T2. In-distribution test: policy summary (10 replicates × 4,000 episodes)",
         t_id_describe(a)),
        ("T3. H1: in-distribution comparisons", t_h1(a)),
        ("T4. Per-replicate in-distribution differences (bp)", t_seeds(a)),
        ("T5. H2: degradation of RL's edge under shift", t_h2(a)),
        ("T6. H3: does the score predict counterfactual regret? (pooled test, 13 conditions)",
         t_h3(a)),
        ("T7. H3: within-condition Spearman ρ of the RCG score", t_within(a)),
        ("T8. H4/H5: gate vs controls (deployment mixture)", t_gates(a["primary"])),
        ("T9. Expected IS by policy and condition (bp, CV estimator)", t_mixture(a["primary"])),
        ("T10. Fallback rates", t_fallback(a["primary"])),
        ("T11. Gate comparison within shift families", t_family(a)),
        ("T12. Robustness: impact models (full re-run) and latency", t_robust(a)),
    ]
    for title, rows in sections:
        out += [f"## {title}", "", *rows, ""]
    return "\n".join(out)
