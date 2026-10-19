# Methodology

This document explains *how* numbers in this repository are produced and how to read them.
The binding protocol for the pre-registered study is
[`research/PREREGISTRATION.md`](../research/PREREGISTRATION.md) (hash-locked). Where this
document and the pre-registration differ, the pre-registration governs. The full results are
in [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md).

## 1. Problem

A desk must sell a parent order of size X (3-12% of expected volume over 30 minutes) in 60
bins of 30 s. Every policy chooses, in each bin, what fraction of X to sell. The cost is
**implementation shortfall** (IS):

    IS = 1e4 · (X·M₀ − Σₜ vₜPₜ) / (X·M₀)      [basis points of arrival notional; lower is better]

An RL policy trained in one market regime may fail in another. The research question:
*can an uncertainty-aware gate decide, at order arrival, when to trust the RL policy and when
to fall back to TWAP?* The quantity that matters is the **counterfactual regret** of trusting
RL on an episode, `IS_RL − IS_TWAP`. Positive regret means RL was the wrong choice for that
order.

## 2. Simulator

A synthetic market (`adexec/sim.py`, frozen) with an exogenous part and an endogenous part:

- **Exogenous**, drawn once per episode from its seed: stochastic volatility (log-AR(1)),
  Student-t returns (ν = 4), a latent short-horizon alpha observed through a noisy signal,
  and spread, depth and volume processes loaded on volatility.
- **Endogenous**, the trader's own impact. Three models are matched in cost at 7.5%
  participation:
  - `sqrt`: concave temporary and cumulative impact, the square-root law (primary);
  - `linear`: Almgren-Chriss;
  - `transient`: sqrt, with half the temporary impact decaying with τ = 120 s.
- **Latency**: fills at t + latency on a Brownian bridge inside the bin.

Validated against stylised facts before any policy was trained
([`SIMULATOR_VALIDATION.md`](SIMULATOR_VALIDATION.md)). The engine's declared impact
exponents are checked empirically in `tests/test_market.py`.

**Common random numbers.** The exogenous market depends only on the episode seed, so every
policy faces the identical market on a given episode. Paired differences therefore remove
market noise, and per-episode regret is exact.

## 3. Policies

| Policy | Inputs | Parameters | Tuned on |
|---|---|---|---|
| TWAP | time, inventory | none | — |
| Almgren-Chriss (static) | time | urgency κ (25-point grid) | TRAIN split, ID |
| ADAPT (heuristic) | the same 9 observable features as RL | 5 weights (coordinate descent) | TRAIN split, ID |
| PPO (RL) | 9 observable features | 64×64 MLP; a ∈ [−1, 1] → sell exp(a·ln 4) × remaining-TWAP pace | RL-training stream, ID only |
| PPO ensemble | 5 members, mean action | — | 10 independent replicates per impact model |
| RCG gate | 15 pre-trade features | regressor of control-variate regret | VALIDATION, leave-one-family-out |

No policy can read latent state (true alpha, true impact coefficient, regime multipliers).

## 4. Distribution shift

There are 12 conditions in 6 families: VOL, LIQ, ALPHA, IMPACT, LATENCY and SIZE, each
moderate and severe. They were frozen in `adexec/config.py` before training, with
magnitudes anchored to documented market episodes. Some shifts are visible in pre-trade data
(LIQ: spreads and depth). Others are hidden: IMPACT raises the impact coefficient with no
change in visible depth, and ALPHA removes the signal's predictive power without changing
its distribution. The deployment **mixture** puts weight 0.5 on ID and 0.5/12 on each shift.

## 5. Data partitions

| Split | Tag | Use |
|---|---|---|
| train | 11 | baseline tuning, OOD reference, gate threshold quantiles |
| val | 22 | gate fitting (LOFO), model selection |
| test | 33 | **confirmatory, once**; guarded by the lock check, every access logged |
| rltrain | 44 | PPO training (study streams 0-2,049) |
| simval | 55 | simulator validation; unit tests |
| benchmark | 66 | **post-study** benchmarks (`benchmarks/`); never used by the study |
| rltrain ≥ 50,000 | 44 | **post-study** PPO training (`execshift train`) |

Streams are disjoint by construction (`SeedSequence([tag, condition, ...])`). A static test
checks every module: only `pipeline/confirmatory.py` and `guard.py` may request the test
split.

## 6. Estimators

- **Realised IS**: what an order actually paid. Used for all risk metrics (SD, VaR, CVaR).
- **Control-variate IS (`IS_cv`)**: realised IS minus a zero-mean price-innovation term,
  built from (quantity fixed at time t) × (innovation after t). Its expectation equals that
  of IS exactly (tower property). It removes 4×-99× of paired-difference variance. It uses
  latent simulator quantities, so it serves only as a *label* and a variance-reduction
  device, never as a policy or gate input. It is the primary estimator for means and
  differences. Every comparison also reports the realised-IS estimate.

## 7. Metrics

One implementation per metric (`adexec/evaluation/metrics.py`), each documented and unit
tested. Where the frozen analysis has its own copy, a test checks that the two agree.

| Metric | Definition | Unit | Read as |
|---|---|---|---|
| implementation shortfall | 1e4 (X·M₀ − Σ vP)/(X·M₀) | bp | cost of execution vs instant sale at arrival mid |
| mean / median cost | sample mean / median of IS | bp | expected / typical cost |
| std | SD of IS (ddof 0, as in the study tables) | bp | dispersion |
| p95 (VaR95) | 95th percentile of IS | bp | a bad-but-not-rare order |
| CVaR95 / CVaR99 | mean IS at or above the 95th / 99th percentile | bp | expected cost of the worst 5% / 1% of orders |
| regret | per-episode IS_policy − IS_reference | bp | > 0: the policy was the wrong choice for that order |
| relative degradation | (value − reference)/\|reference\| | — | > 0: worse than the reference |
| difference-in-differences | mean(a−b \| shift) − mean(a−b \| ID) | bp | the study's H2 degradation D_c |
| win rate | P(a < b) over paired episodes, ties ½ | — | a rate, not an effect size |
| Spearman ρ | rank correlation of score and regret | — | > 0: score ranks harmful orders higher |
| AUROC | P(score of a harmful order > score of a harmless one) | — | 0.5 = chance |
| AUPRC | average precision for harmful orders | — | compare with the base rate, not 0.5 |
| risk-coverage | mean regret when trusting RL on the lowest-score fraction | bp | selective-prediction curve |

## 8. Statistics

**Units.** The two random factors are RL training replicates (r = 1..10) and market episodes,
and they are crossed. Steps within an episode are never units.

| Situation | Procedure | Module |
|---|---|---|
| study: replicates × episodes, paired by episode | two-way crossed ("pigeonhole") bootstrap, B = 10,000, seed 20261001; mixture-weighted | `stats.crossed_bootstrap` (frozen); `evaluation.statistics.crossed_comparison` |
| post-study benchmark: one replicate, paired episodes | percentile bootstrap over episodes, B = 2,000, seed 0 | `evaluation.statistics.paired_comparison` |
| independent samples (no shared episodes) | independent-samples bootstrap; Cohen's d | `evaluation.statistics.unpaired_comparison` |
| multiple comparisons | Holm within each hypothesis family | `stats.holm` |
| replicate-level check | Wilcoxon signed-rank over the 10 replicate means | `stats.seed_level_wilcoxon` |

**Never significance without effect size.** Every comparison object carries the estimate
(bp), the 95% CI, a p-value, a standardised effect size (d_z), the practical threshold and
its source, and a verdict:

| Verdict | Condition |
|---|---|
| `a practically better` | CI below 0 **and** estimate < −threshold |
| `b practically better` | mirror image |
| `significant but below threshold` | CI excludes 0, \|estimate\| < threshold. **The study's H4 is here: −0.013 [−0.024, −0.004] bp vs 0.25 bp** |
| `practically equivalent` | CI inside (−threshold, +threshold) |
| `inconclusive` | otherwise |

**The practical threshold** (smallest effect size of interest, SESOI) is **0.25 bp**,
pre-registered and hash-locked (`analysis.SESOI`). It is about 3.5% of TWAP's
in-distribution cost. The configuration system refuses any other threshold unless the config
also names a `threshold_source`, so an exploratory threshold is always labelled as one.

**Post-study benchmark intervals understate uncertainty about RL.** They cover episode
sampling only. RL training-seed variation is not inside them, unless several replicates are
listed as separate policies (the `seeds` suite does this). The confirmatory study's crossed
bootstrap includes both factors.

## 9. Gate fitting without leakage

Every gate threshold and regressor is fitted on validation data under **leave-one-family-out**
(LOFO): the gate applied to shift family f never saw family f. A "seen-family" variant is
reported as exploratory. The matched-random control (A10) falls back at exactly RCG's rate
in each condition and replicate, so it isolates *which* orders are gated from *how many*.

## 10. What counts as a finding

| Label | Meaning | Where |
|---|---|---|
| **PRE-REGISTERED CONFIRMATORY** | H1-H6, decided by the locked rules on the single test run | `results/authoritative.json`, `docs/FINAL_RESEARCH_REPORT.md` |
| **EXPLORATORY (in the study)** | analyses the pre-registration declared exploratory (ablation gates, seen-family gate) | same files, marked |
| **POST-STUDY DESCRIPTIVE** | re-summaries of the frozen confirmatory outputs; no new tests | `results/post_study/`, `figures/supplementary/` |
| **POST-STUDY / EXPLORATORY** | new runs on new episodes (benchmarks, new scenarios, new training) | `runs/<id>/` (git-ignored), [`BENCHMARKS.md`](BENCHMARKS.md) |

A post-study result never changes a pre-registered verdict, however it comes out.
