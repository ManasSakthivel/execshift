# Future work

The study's negative result is specific. A pre-trade gate built on generic uncertainty
cannot recover much cost, because (a) the room for a fallback to help is small under
realistic impact models, (b) the worst failure mode, concept shift, is invisible before the
order starts, and (c) out-of-distribution distance and ensemble disagreement are not regret.
Each direction below attacks one of these causes.

**Rule for all of it:** none of this is part of the pre-registered study. Each item needs
its own pre-registration (new hypotheses, new SESOI if any, a new locked test stream) before
any confirmatory claim. The platform makes that cheap: new scenarios, policies, estimators
and suites plug in without touching the frozen core ([`ARCHITECTURE.md`](ARCHITECTURE.md)
§10). Nothing here has been implemented, to keep the study uncontaminated.

Priority reflects the expected information per unit of effort.

## 1. Direct regret prediction

- **Current limitation.** The RCG gate regresses regret on generic features (market summary,
  OOD distance, ensemble disagreement, shadow replay). The result is ρ = +0.108, AUROC 0.542.
  The features were designed to detect *shift*, not *harm*.
- **Proposed method.** Learn a model of the conditional regret distribution directly:
  quantile regression or a distributional model of IS_RL − IS_TWAP. Use features chosen for
  their causal link to RL's failure modes, for example RL's sensitivity to the alpha signal,
  the predicted participation path versus the visible depth, and the cost-model mismatch
  implied by shadow replay. Train it on a large, deliberately diverse pool of shifted
  markets, *disjoint* from the evaluation shift families.
- **Required data.** Synthetic: a new training stream with a broad shift distribution, as
  many as needed. Real: as for §6.
- **Experiment design.** Leave-one-family-out exactly as in the study. Primary endpoint:
  AUROC for regret > 0 and mixture IS of the induced gate against always-RL, with the 0.25 bp
  SESOI. A pre-declared comparison against RCG and matched random switching.
- **Potential failure modes.** Regret is dominated by future price noise. Against realised
  regret, ρ halves, and the control variate removes 4-99× of paired variance. No decision-time
  model can predict that part. The ceiling is the headroom (0.065 bp under sqrt impact), so
  even a perfect predictor cannot pass H4 there. That is worth showing, as it bounds the
  whole research line.

## 2. Online mid-order intervention

- **Current limitation.** The gate decides once, at arrival. Alpha decay (ALPHA-sev, where
  RL loses) leaves the pre-trade window unchanged, so the gate falls back on 0.1% of those
  orders.
- **Proposed method.** Re-decide every k bins using in-episode evidence: the realised
  signal-return correlation, fill prices versus the cost model, and RL's realised versus
  expected shortfall. Formulate it as a sequential test (CUSUM or a likelihood ratio on the
  signal's predictive power) or as a learned switching policy. The `ExecutionPolicy`
  interface supports this directly (`reset` + per-step `act`), and the gate's per-episode
  mask would become per-step.
- **Required data.** Synthetic episodes with *within-episode* regime changes, a scenario
  type the current suite does not have (add as `... (NEW)` scenarios).
- **Experiment design.** Arrival-only gate versus online gate versus always-RL, on the
  study's mixture and on a new set of within-episode change points. Primary endpoint:
  mixture IS. Secondary: detection delay versus false-switch rate.
- **Potential failure modes.** Switching late in the order recovers little, because most
  inventory is gone. Switching on noise adds cost. Alpha decay may be detectable only after
  most of its cost is paid.

## 3. Conformal risk prediction

- **Current limitation.** The gate's thresholds are point decisions with no coverage
  guarantee. The study shows that calibration fails under shift.
- **Proposed method.** Conformal risk control (Angelopoulos et al. 2024) or adaptive
  conformal inference (Gibbs & Candès 2021) on regret. Trust RL only when the conformal upper
  bound on regret is below zero, with guaranteed expected-regret control under
  exchangeability. Use weighted conformal methods to handle covariate shift.
- **Required data.** A calibration set per deployment regime, or an online stream for
  adaptive conformal inference.
- **Experiment design.** Measure empirical regret control (does realised mean regret stay
  below the target α?) and efficiency (fallback rate), in- and out-of-family. Compare with
  RCG and with the always-RL / always-TWAP extremes.
- **Potential failure modes.** Concept shift breaks exchangeability, and the guarantee with
  it. The efficient bound may simply recommend always-TWAP. That would be a valid,
  informative answer.

## 4. Learned market-regime representations

- **Current limitation.** Hand-built pre-trade summaries (eight statistics) and a
  Mahalanobis distance. OOD distance fires on volatility shifts, which are harmless for RL.
- **Proposed method.** Learn a representation of the pre-trade window (contrastive or
  self-supervised on market paths) in which distance tracks *policy-relevant* change.
  Supervise it with regret, or with RL's value-prediction error, rather than with feature
  novelty.
- **Required data.** Large unlabelled synthetic and real pre-trade windows; regret labels
  from simulation.
- **Experiment design.** Replace the market-summary features in RCG with the
  representation, holding everything else fixed (LOFO, thresholds, SESOI). Ablate
  supervised against self-supervised training.
- **Potential failure modes.** Representations overfit the simulator's specific shift
  mechanisms and do not transfer. Hidden concept shift stays invisible in any pre-trade
  representation.

## 5. Cross-market validation

- **Current limitation.** One synthetic, liquid-equity-like market.
- **Proposed method.** Re-parameterise the simulator for distinct microstructures: small-cap
  equities (wide spreads, thin books), futures (deep books, low spreads) and FX (24 h,
  fragmented). Rerun the benchmark suites and the confirmatory design per market.
- **Required data.** Published parameter ranges per asset class; ideally TAQ / futures
  tick data for calibration.
- **Experiment design.** Pre-register per-market hypotheses. The key quantity is the
  *headroom* per market: if it stays small everywhere, the gating research line is bounded.
- **Potential failure modes.** The calibration is only as good as the parametric model.
  Market-specific features, such as auctions and maker-taker fees, are not in the engine.

## 6. Real limit-order-book validation

- **Current limitation.** No real data. Nothing here transfers to a live venue.
- **Proposed method.** (a) Replay-based counterfactual evaluation on LOB data
  (LOBSTER-style message data), with an impact model for the trader's own orders.
  (b) Measure the fallback headroom empirically: how often, and by how much, a learned
  execution policy underperforms TWAP across real regimes.
- **Required data.** Level-2/3 LOB message data (licensed); a historical execution log if
  available.
- **Experiment design.** Replay cannot show other participants' reactions, so pair it with
  conservative impact assumptions and report results per impact model, as this study does.
  Pre-register the headroom measurement first. It decides whether gating is worth pursuing
  on real data at all.
- **Potential failure modes.** Replay bias (no market reaction to our orders), licensing
  limits on publishing results, and non-stationarity across sample periods.
