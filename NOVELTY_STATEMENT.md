# Novelty statement

> **Question:** What does this project add that existing work does not already establish?

The evidence behind this statement is [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md).
We make no claim on the grounds that "nobody has done exactly this". Each claim below names
the closest prior work and the specific difference.

## What is NOT new, and we do not claim

1. **Falling back from a learned policy to a heuristic when the learned policy is uncertain.**
   This was proposed for deep-RL systems by Rotman, Schapira & Tamar (2020, adaptive
   bitrate). It is the architecture of Simplex and Neural Simplex (Sha 2001; Phan et al.
   2020) and of Recovery RL (Thananjeyan et al. 2021). For forecast-driven execution it was
   studied by Irshad & Biswas (2026), who gate a return forecast with a conformal interval
   and default to VWAP tracking.
2. **RL beating TWAP/AC in-distribution** (Nevmyvaka et al. 2006; Hendricks & Wilcox 2014;
   Ning et al. 2021; Lin & Beling 2020; Fang et al. 2021).
3. **RL execution policies degrading or overfitting off-distribution** (Zhang et al. 2023;
   Hambly et al. 2023).
4. **OOD and uncertainty scores for neural networks and RL agents** (Lee et al. 2018;
   Lakshminarayanan et al. 2017; Sedlmeier et al. 2020; Nasvytis et al. 2024).
5. **Learning to defer** (Madras et al. 2018) and **per-instance algorithm selection**
   (Rice 1976). RCG is an instance of both.

## What this project contributes

**C1. A regret-centred formulation of trusting RL in execution.**
- Prior gates are scored on whether their signal flags OOD inputs (Rotman 2020; Sedlmeier
  2020) or on forecast-interval coverage (Irshad & Biswas 2026).
- We define the gate's target as the *counterfactual regret* of the deployed RL policy
  relative to the fallback, on the same market (IS_RL − IS_TWAP, measured under common
  random numbers). We evaluate every candidate signal by whether it predicts that regret:
  Spearman correlation, AUROC/AUPRC for harmful episodes, calibration, and risk-coverage.
- The proposed gate (RCG) falls back exactly when predicted regret is positive. This is the
  Bayes decision for expected cost, so no threshold is tuned.
- *Closest prior work:* Rotman et al. (2020) evaluate defaulting performance but select
  signals by OOD-ness; Recovery RL predicts constraint violation, not cost regret.

**C2. A control-variate regret label.**
- Realised per-episode regret is dominated by future price noise, which is unpredictable at
  the decision time. Training on it teaches the gate noise.
- We subtract the zero-mean price-innovation component, which is exact by the tower
  property. This gives an unbiased, roughly 2× lower-variance estimate of each episode's
  expected regret.
- It is used for gate labels and as the primary estimator of expected IS. Realised-IS
  results are reported for every comparison.
- We did not find this device used for training or evaluating deferral or fallback gates in
  the RL-execution literature.

**C3. A pre-registered, leave-one-shift-family-out benchmark for selective intervention in execution.**
- Twelve shift conditions in six families (volatility, liquidity drought, alpha decay,
  hidden impact, latency, order size), anchored to documented market episodes and frozen
  before any policy was trained.
- Every gate is fitted with the *test family held out*, so each shifted test condition is an
  unseen kind of shift.
- The benchmark separates covariate shifts, which are visible in observations, from concept
  shifts, which are not (alpha decay, hidden impact, latency). The second kind is where
  OOD-based gating is expected to fail.
- *Closest prior work:* Zhang et al. (2023) study generalisation of a single execution
  policy; Nasvytis et al. (2024) criticise artificial RL-OOD benchmarks but do not address
  execution.

**C4. Matched-budget controls and a design that tests whether the gate's decisions beat chance.**
- The gate is compared with a rule gate (A9), a policy blend (A3), and random switching with
  the same fallback rate, per condition and replicate (A10, evaluated in exact expectation).
- It is also compared with always-TWAP, OOD-only and ensemble-only gates.
- The analysis uses ≥10 independent RL training replicates, each a 5-member ensemble, and a
  crossed bootstrap over replicates × episodes.
- *Closest prior work:* the execution studies listed above use 1–5 seeds and no
  matched-budget random control.

**C5. Empirical findings in a validated synthetic market,** reported whether positive or
negative: the size of RL's in-distribution edge over tuned baselines, which natural shifts
erode it, whether uncertainty predicts regret, and whether gating beats the controls. They
are stated, with their limits, in `docs/FINAL_RESEARCH_REPORT.md` and
`FINAL_STANFORD_READINESS_REPORT.md`.

## Scope limits on the novelty claim

- Every result is from a **synthetic** simulator (validated against stylised facts) and
  holds conditional on its impact models. Nothing here is evidence about live markets.
- C1–C4 are methodological contributions to *how* the question is posed and tested. If the
  confirmatory results are negative, the contribution is the negative result plus the
  benchmark, not a working method.
