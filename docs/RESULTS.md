# Results reference

This page is a curated index of the study's results. Every value is copied from a committed
artifact and carries its source. The exhaustive tables are in
[`results/tables.md`](../results/tables.md) (T1-T12). The narrative is in
[`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md). Definitions of every metric are in
[`METHODOLOGY.md`](METHODOLOGY.md) §7.

All data are synthetic. These numbers describe one simulated market and are not evidence of
real trading performance.

## 0. Which results are authoritative

| Artifact | Status |
|---|---|
| `results/authoritative.json` | **Authoritative.** Written once by the confirmatory run (`execshift reproduce --stage confirm`). SHA-256 pinned in `tests/test_reproducibility.py`. |
| `results/authoritative.reference.json` | Byte-identical reference copy. `execshift reproduce --verify` compares the two. |
| `results/tables.md`, `docs/FINAL_RESEARCH_REPORT.md` | Rendered from `authoritative.json` by `execshift report --study`. CI fails if they drift. |
| `results/figure_data.json` | Per-condition CIs and gate curves for figures 2-10, computed from the raw outputs. Pinned. |
| `results/post_study/supplementary_data.json` | POST-STUDY DESCRIPTIVE reductions for figures S4-S7. No new tests. |
| `results/fast/` | Toy-scale smoke pipeline output (git-ignored). **Not study results.** |
| `runs/<id>/` | POST-STUDY / EXPLORATORY benchmark runs on a separate episode stream (git-ignored). See [`BENCHMARKS.md`](BENCHMARKS.md). |

There is one set of confirmatory results, with no competing versions. A clean-clone
reproduction on 2026-10-02 regenerated `authoritative.json` byte-for-byte
([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §10).

**Common conventions.** IS is implementation shortfall in basis points of arrival notional.
Lower is better. A difference `A - B < 0` favours A. The primary estimator is the
control-variate IS. Brackets are 95% crossed (replicate × episode) bootstrap CIs. The
practical threshold (SESOI) is 0.25 bp. Unless stated otherwise: square-root impact, 10 RL
replicates, each a 5-member PPO ensemble, on the confirmatory test split.

## 1. H1: does RL beat tuned classical baselines in-distribution?

| | |
|---|---|
| Compared | RL (5-member ensemble) vs TWAP, tuned Almgren-Chriss (AC), tuned ADAPT heuristic |
| Conditions | ID test split, 4,000 episodes × 10 replicates |
| Metric | difference in expected IS (bp) |
| Result | RL - TWAP = -2.45 [-2.57, -2.34]; RL - AC = -2.44 [-2.55, -2.34]; RL - ADAPT = -0.18 [-0.29, -0.07]. 10/10 replicates favour RL over TWAP. |
| Verdict | **Supported** (pre-registered rule: both TWAP and AC differences < -0.25 bp, Holm-significant) |
| Source | `authoritative.json` → `primary/H1`; tables T2-T4 |
| Interpretation | RL learns a real in-distribution edge. A tuned heuristic that uses the same features gets almost all of it: the RL - ADAPT gap is below the SESOI, and in realised IS it is +0.10 [-0.16, +0.37]. |
| Limitation | The edge depends on the simulator's assumed alpha signal. |

## 2. H2: does RL's edge degrade under distribution shift?

| | |
|---|---|
| Compared | RL - TWAP in each shifted condition vs the same difference in ID (difference-in-differences D_c) |
| Conditions | 12 shift conditions, 1,000 episodes × 10 replicates each |
| Result | D_c > SESOI and Holm-significant in 8 of 12 conditions. RL is worse than TWAP outright in LIQ-sev (+0.97 bp) and ALPHA-sev (+0.59 bp). The edge *grows* under volatility shifts (VOL-sev D_c = -4.10 [-4.69, -3.50]). |
| Verdict | **Supported** |
| Source | `primary/H2`; table T5; figure 4 |
| Interpretation | Shift hurts RL in specific, identifiable regimes rather than uniformly. |

## 3. H3: does the gate's score predict counterfactual regret?

| | |
|---|---|
| Compared | RCG predicted regret, plus exploratory scores: OOD distance, ensemble disagreement, shadow regret |
| Conditions | all 13 test conditions pooled; 16,000 episodes per replicate; metric per replicate, then averaged over 10 replicates |
| Metric | Spearman ρ(score, regret); AUROC for regret > 0 (base rate 0.301) |
| Result | RCG ρ = +0.108 [+0.072, +0.138], AUROC 0.542 [0.519, 0.559]. OOD distance ρ = +0.027, AUROC 0.541. Ensemble action std ρ = -0.016, AUROC 0.481. Within any single condition, RCG's ρ is at most +0.103. |
| Verdict | **Supported by the pre-registered rule** (both CIs exclude chance), **practically weak** |
| Source | `primary/H3`; tables T6-T7; figures 5, S6, S7 |
| Independent check | ρ and AUROC were recomputed from `results/raw/` with SciPy and scikit-learn and match to every printed digit. `tests/test_reporting.py` checks that S7's scikit-learn AUROCs equal the study's rank-based values. |
| Interpretation | The score ranks harmful orders above harmless ones only slightly more often than chance. AUROC measures *ranking*, not *calibration*: it says nothing about whether predicted regret has the right magnitude. ρ is a correlation. It does not show that uncertainty causes, or explains, regret. |

## 4. H4: does the gate beat always trusting RL? (central claim)

| | |
|---|---|
| Compared | RCG-gated policy vs always-RL |
| Conditions | deployment mixture: 50% ID, 50% spread equally over the 12 shifts |
| Result | RCG - always-RL = -0.013 [-0.024, -0.004] bp, Holm p = 0.0054 |
| Verdict | **Not supported.** The difference is statistically detectable but about 20× smaller than the 0.25 bp threshold. |
| Source | `primary/H45/H4`; table T8; figure 8; `execshift inspect --claim H4` |
| Interpretation | A gate that can only tell *conditions* apart is bounded by the *headroom*: the mixture-weighted cost RL incurs in the conditions where it loses to TWAP. Under square-root impact that headroom is 0.065 bp, below the SESOI. An unachievable per-order oracle, which knows each order's regret in advance, beats always-RL by 0.31 bp (mixture IS 6.022 vs 6.332, table T9). So the SESOI was reachable in principle only by a gate that discriminates *within* conditions, and the measured within-condition ρ is at most +0.103. |

## 5. H5: does the gate beat simple controls?

| Comparison (deployment mixture) | Δ expected IS, bp | Holm p |
|---|---|---|
| RCG - A3 blend | -0.344 [-0.365, -0.323] | <0.0001 |
| RCG - A9 rule gate | -0.443 [-0.481, -0.404] | <0.0001 |
| RCG - A10 random (matched fallback rate) | -0.005 [-0.010, -0.001] | 0.0276 |
| RCG - TWAP | -2.167 [-2.242, -2.092] | <0.0001 |

Verdict: **supported under the primary impact model only** (see H6). Source: `primary/H45`,
table T8. The matched-random control is the informative one. It falls back on exactly as
many orders as RCG in each condition and replicate, so the -0.005 bp difference measures the
value of *which* orders RCG picks. That value is tiny.

## 6. H6: robustness to impact model and latency

| Setting | RL - TWAP (ID) | H4: RCG - RL | RCG - A9 | H3 ρ |
|---|---|---|---|---|
| sqrt (primary) | -2.45 [-2.57, -2.34] | -0.013 [-0.024, -0.004] | -0.443 [-0.481, -0.404] | +0.108 |
| linear | -1.29 [-1.39, -1.18] | -0.608 [-0.753, -0.468] | -0.018 [-0.050, +0.016] | +0.328 |
| transient | -2.82 [-2.94, -2.70] | +0.000 [-0.000, +0.001] | -0.635 [-0.675, -0.595] | -0.013 |
| latency 5 s (sqrt) | — | -0.014 [-0.025, -0.005] | -0.425 [-0.463, -0.387] | — |

Verdict: **not supported.** The conclusions hold across latency (0, 1 and 5 s give almost
identical results) but not across impact models. Under linear impact the gate gains 0.608 bp,
more than the SESOI, but the simple rule gate does statistically as well. Under transient
impact the gate almost never acts: at most 0.6% of orders, in LIQ-sev. Source: `impact_models/*`, `latency/*`; table T12; figure 9,
figure S3.

**Headroom vs realised gain.** The condition-level oracle headroom is 0.065 (sqrt),
0.788 (linear) and 0.011 bp (transient). The gate realises 0.013, 0.608 and 0.000 bp
(computed by `reporting.study_report.headroom` from `authoritative.json`). The value of
gating tracks the headroom far more than any property of the gate.

## 7. Where the gate acts

| Condition | RCG fallback rate | RL - TWAP in condition |
|---|---|---|
| LIQ-sev (visible: spreads, depth) | 36.8% | +0.97 bp |
| ALPHA-sev (hidden: signal decay) | 0.1% | +0.59 bp |
| VOL-sev | 0.0% | -6.55 bp |

Source: tables T5 and T10; figure S2. The gate reacts to shifts that are visible in
pre-trade data and misses the hidden one, because the signal's *distribution* does not change
when it stops predicting returns. OOD distance fires hardest on volatility shifts (58.6%
fallback in VOL-sev), which is exactly where RL is strongest.

## 8. Tail risk (exploratory)

In-distribution CVaR95 of realised IS: RL 79.1 bp, ADAPT 64.9 bp, TWAP 79.4 bp (table T2;
figure S4). RL and ADAPT have practically tied means. The study's objective is risk-neutral,
so this is a descriptive observation, not a tested hypothesis.

## 9. What these results do not establish

- Anything about real markets, live trading or profitability.
- That uncertainty-based gating fails in general. They show it gains little when the
  fallback headroom is small, and that it misses shifts with no pre-trade signature, in this
  simulator.
- That RL is generally superior to classical execution. It beats TWAP and AC here, under
  this simulator's alpha signal, and ties a tuned heuristic.
- Any result from `runs/` or `results/fast/`. Post-study benchmarks are exploratory and
  cannot change a pre-registered verdict.
