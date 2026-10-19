# Limitations

What this repository can and cannot support. The study's own limitations section is §19 of
[`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md). This page repeats its substance and
adds the limitations of the engineering platform and the documentation defects found in
audits.

## 1. Scope of evidence

- **All data are synthetic.** Nothing here is evidence about live trading, profitability or
  deployability. No result transfers to a real venue without real-data validation
  ([`FUTURE_WORK.md`](FUTURE_WORK.md) §6).
- **One market model.** A single generic, liquid-equity-like asset with parametric impact.
  No calibration to an instrument or sample period.
- **Conclusions depend on the impact model.** Under sqrt and transient impact, RL almost
  never loses to TWAP, so a fallback gate has almost nothing to gain (headroom 0.065 and
  0.011 bp). Under linear impact the headroom is 0.788 bp, and the gate helps materially.
  Which model is right for a given market is an empirical question this repository cannot
  settle.

## 2. Simulator

- Marketable orders only: no limit-order placement, queue position or tick size.
- No fees, no market-wide feedback beyond the impact model, and no reaction of other
  participants to our orders.
- The volatility memory is exponential (log-AR(1)), not power-law. There is no intraday
  seasonality and no leverage effect (`results/simulator_validation.json`, "not modelled").
- The alpha signal is an assumed stylised predictor. RL's in-distribution edge depends on it.

## 3. Study design

- **The gate decides once, at arrival.** Mid-order intervention was not studied. The
  hidden concept shifts that pre-trade gates miss (alpha decay) might be detectable from
  in-episode evidence.
- **The fallback is TWAP.** ADAPT has lower tail risk than RL and nearly the same mean, but
  it was not studied as a fallback.
- **The bounded action space** ([¼, 4] × TWAP pace) limits how badly RL can lose, and
  plausibly contributes to the small headroom. It was not ablated.
- **Objective is risk-neutral.** RL has higher tail risk than ADAPT: CVaR95 79.1 vs
  64.9 bp in-distribution. The post-study `risk` suite finds ADAPT with the lowest CVaR95 in
  11 of 13 conditions.
- **Search budgets:** 16 PPO configurations, ≤ 361 ADAPT evaluations, 25 AC values. Ten
  replicates per impact model.

## 4. Documentation defects found by audit (not corrected in frozen files)

- **S1. The ADAPT tuning grid in the pre-registration is stale.** `PREREGISTRATION.md`
  §6.1 says "a 17-point grid in [−2, 2]". The locked `tuning.py` and the results use a
  25-point grid on [−3, 3], widened before any RL evaluation (`docs/BASELINES.md`). §11
  "Deviations" does not record it. The code that ran is the locked code, so the result is
  internally consistent. A wider grid can only strengthen the baseline, which makes H1
  harder to pass, not easier. Because the pre-registration is hash-locked, the discrepancy
  is recorded here and in `ARCHITECTURE_AUDIT.md` §7, not edited.
- **S2. The pre-lock timeline is not independently corroborated by git.** All development
  history is in one commit, 7 s before the lock commit. The order of the timeline (shifts
  frozen before training, development on validation only) rests on the authors' account.
  The test split's integrity has stronger support: a hash-gated guard, an access log in
  which all 780 entries post-date the lock, and a static test. The log itself is
  append-only by convention, not tamper-evident.
- **S3. Verifier coverage.** The pre-declared verifier checks 1,063 values and decisions
  (estimates, CI bounds, booleans), not the H3 point metrics. The post-study hardening adds
  an informational comparison of every other numeric leaf and a `--strict` mode. The
  pre-declared tolerance is unchanged.

## 5. Platform limitations

- **Post-study benchmark intervals cover episodes only.** With one RL replicate they omit
  training-seed variance. In LIQ-sev this made replicate 0 look inconclusive while the
  10-replicate mean reproduces the study's loss ([`BENCHMARKS.md`](BENCHMARKS.md) §3.4).
  Use the `seeds` suite, or list replicates, for claims about RL.
- **The learned RCG gate is not in the benchmark suites.** It needs validation-split gate
  features and LOFO fitting, which live in the confirmatory pipeline. The `gate` suite rolls
  out its single-signal components (OOD, ensemble, shadow) instead.
- **PPO checkpoints are not committed** (22 MB; weight hashes in `results/rl_manifest.json`).
  RL benchmarks need `make train` (~76 min on 14 cores for all 150 members) first. CI tests
  RL code paths with a tiny trained fixture, never with the study's checkpoints.
- **Raw per-episode outputs are pickles** (343 MB, git-ignored, local only). Load only files
  this pipeline produced.
- **Cross-platform byte-identity is unverified.** Same-platform reproduction is
  byte-identical. Linux x86-64 runs are expected to differ in PPO weights at floating-point
  level. The verifier's tolerances are designed for this but have not been exercised on a
  second platform.
- **mypy covers the platform layers only.** The frozen study modules cannot be edited to
  satisfy a type checker without breaking the lock.
- **Python 3.12 only.** `requirements-lock.txt` pins wheels that need 3.12 (e.g. contourpy
  1.4.0), so CI tests 3.12 only. Byte-identity claims apply only to the recorded environment
  (3.12.14).
