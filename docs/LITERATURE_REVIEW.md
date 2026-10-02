# Literature review

**Scope.** This review asks whether the question this project studies is already answered.
The question: *can an uncertainty-aware gate decide when to trust an RL execution policy
under distribution shift, and fall back to a classical schedule when RL's expected regret
is positive?*

**Method.** We searched Google Scholar, arXiv, SSRN and publisher pages (Oct 2026) using
combinations of: reinforcement learning, optimal execution, trade execution, fallback,
default policy, uncertainty, out-of-distribution, selective prediction, conformal, safe
policy improvement, shielding, simplex architecture, regime, distribution shift.
For each paper we retrieved the abstract and, where it was accessible, the full text.

**Verification caveat.** Citations marked † were confirmed against an abstract or
publisher page during this review. The rest are standard references that we are confident
of but did not re-verify page by page.

Each entry gives: **Q**, the research question; **M**, the method; **D**, the data or
simulator; **E**, the evaluation; **L**, the limitations; and **→**, the relationship to
this project.

---

## 1. Classical optimal execution (the fallback and the baselines)

**Bertsimas & Lo (1998)**, *Optimal control of execution costs*, J. Financial Markets 1(1).
- Q: How to minimise the expected cost of executing a block under linear price impact.
- M: Dynamic programming.
- D: Analytical model.
- E: Closed form. Without information, the optimal schedule is to split the order evenly (TWAP).
- L: Risk-neutral; linear impact.
- → Why TWAP is the natural risk-neutral fallback.

**Almgren & Chriss (2001)**, *Optimal execution of portfolio transactions*, J. Risk 3(2).
- Q: How to trade off expected cost against cost variance.
- M: Mean-variance optimisation with linear temporary and permanent impact.
- D: Analytical model.
- E: Efficient frontier; sinh-shaped trajectories.
- L: Static (open-loop) schedules; linear impact; parameters assumed known.
- → Our AC baseline. Under our risk-neutral primary endpoint, tuned AC collapses to TWAP, as the theory predicts. We report this rather than hide it.

**Obizhaeva & Wang (2013)**, *Optimal trading strategy and supply/demand dynamics*, J. Financial Markets 16(1).
**Gatheral (2010)**, *No-dynamic-arbitrage and market impact*, Quantitative Finance 10(7).
- Q: How do transient (decaying) impact and order-book resilience change optimal execution, and which impact models are free of price manipulation?
- M: A propagator model with exponential resilience.
- E: Gatheral shows that nonlinear impact must decay to avoid manipulation.
- → Our "transient" impact model. Our concave cumulative impact is safe from round-trip manipulation because the agent can only sell.

**Almgren, Thum, Hauptmann & Li (2005)**, *Direct estimation of equity market impact*, Risk.
**Tóth et al. (2011)**, *Anomalous price impact and the critical nature of liquidity*, PRX.
**Donier, Bonart, Mastromatteo & Bouchaud (2015)**, Quantitative Finance.
- Q: How does metaorder impact scale with size?
- D: Proprietary metaorder data from Citigroup (Almgren et al.) and CFM (Tóth et al.).
- E: Impact grows roughly as σ·√(Q/V), with exponent ≈ 0.5–0.6 and prefactor Y = O(1).
- L: Averages over many orders; individual metaorders are noisy.
- → Calibration and validation targets for our simulator. Our first simulator version **failed** this check (slope 0.87); we fixed it (see `SIMULATOR_VALIDATION.md`).

**Cartea & Jaimungal (2016)**, *Incorporating order-flow into optimal execution*, Math. Financial Econ. 10(3).
**Cartea, Jaimungal & Penalva (2015)**, *Algorithmic and High-Frequency Trading*, CUP.
- Q: How to execute optimally when order flow predicts short-term drift.
- M: Stochastic control with an order-flow signal.
- E: Closed forms; trading speeds up when the signal is adverse.
- → The economic mechanism behind both ADAPT's and RL's in-distribution edge.

**Cartea, Donnelly & Jaimungal (2017)**, *Algorithmic trading with model uncertainty*, SIAM J. Financial Math. 8(1).
- Q: How to execute when the model itself is ambiguous.
- M: Robust control: the agent optimises against the worst model within a relative-entropy penalty.
- E: Ambiguity aversion produces more conservative, closer-to-TWAP strategies.
- L: Robustness is built into one policy rather than detected per episode.
- → An alternative to gating. Robust-control designs pay their price in every episode; a selective gate pays only when it falls back.

## 2. RL for trade execution

**Nevmyvaka, Feng & Kearns (2006)**†, *Reinforcement learning for optimized trade execution*, ICML.
- Q: Can RL improve the limit-order placement of a fixed-size order?
- M: Tabular Q-learning over discretised private state (time, inventory) and market state (spread, volume imbalance, etc.).
- D: 1.5 years of millisecond-level NASDAQ INET order-book data for three stocks; historical replay.
- E: IS relative to submit-and-leave. They report large relative improvements that grow with richer market-state variables.
- L: Replay cannot capture how the market reacts to the agent's own orders; small tabular state space.
- → The founding result: market-state features carry execution-relevant information. That is also why shifts in those features matter.

**Hendricks & Wilcox (2014)**†, *A reinforcement learning extension to the Almgren-Chriss framework for optimal trade execution*, IEEE CIFEr.
- Q: Can Q-learning improve an AC schedule using microstructure features?
- M: Q-learning that adjusts AC volume trajectories.
- D: South African equity data.
- E: Reported IS improvement over AC on the order of 10%.
- → RL as a *residual* on a classical schedule. Our action space (a multiplier on the remaining-TWAP pace) follows the same idea.

**Ning, Lin & Jaimungal (2021)**, *Double deep Q-learning for optimal execution*, Applied Mathematical Finance 28(4).
**Lin & Beling (2020)**, *An end-to-end optimal trade execution framework based on proximal policy optimization*, IJCAI.
- Q: Can deep RL beat TWAP/AC on real order-book data?
- M: DDQN; PPO with LSTM.
- D: Historical US equity data; replay.
- E: Better IS than TWAP and AC on most stocks.
- L: Replay with stylised impact; few seeds; no distribution-shift or fallback analysis.
- → Motivates H1. We require ≥10 independent replicates, which these studies did not use.

**Fang et al. (2021)**, *Universal trading for order execution with oracle policy distillation*, AAAI.
- Q: How to learn from noisy market data.
- M: A teacher with access to future information distils its policy into a student.
- D: Chinese A-share minute bars.
- E: Better than TWAP/VWAP and DRL baselines.
- → Shows that the in-distribution edge is achievable; it does not address deployment under shift.

**Moallemi & Wang (2022)**, *A reinforcement learning approach to optimal execution*, Quantitative Finance.
- M: Treats execution as optimal stopping and learns from historical data.
- E: Reports substantial improvements over benchmarks on NASDAQ data.
- → Another strong in-distribution result, with no fallback analysis.

**Zhang et al. (2023)**†, *Towards generalizable reinforcement learning for trade execution*, IJCAI.
- Q: Why do RL execution policies overfit?
- M: Frames execution as offline RL with dynamic context; derives generalisation bounds; learns compact context representations.
- D: High-fidelity LOB simulator built from historical data.
- E: Their representations generalise better than prior methods.
- L: They improve *generalisation* of a single policy; they do not decide *when not to trust it*.
- → The closest analysis of *why* RL execution degrades. Complementary to gating.

**Macrì & Lillo (2024)**†, *Reinforcement learning for optimal execution when liquidity is time-varying*, Applied Mathematical Finance 31.
- Q: Can DDQL learn optimal execution when impact parameters vary and are latent?
- M: DDQL in an AC framework with deterministic and stochastic impact dynamics.
- E: Matches the analytical optimum where one exists; beats approximations elsewhere.
- → Supports our modelling choice that the impact coefficient is latent (our hidden IMPACT shift).

**Karpe et al. (2020)**, ICAIF; **Byrd, Hybinette & Balch (2020)**, *ABIDES*, ICAIF; **Amrouni et al. (2021)**, *ABIDES-Gym*, ICAIF.
- Agent-based LOB simulators used to train and test execution agents.
- → The alternative simulator family. We use a parametric simulator instead: it is fast enough for 150 PPO members and gives exact counterfactual regret through common random numbers. The price is less microstructure realism (see Limitations).

**Hambly, Xu & Yang (2023)**†, *Recent advances in reinforcement learning in finance*, Mathematical Finance 33(3).
- A survey. For execution, it notes the field's reliance on simulators and historical replay, and that robustness to market changes and model misspecification is open.
- → Frames the gap that our question addresses.

**Irshad & Biswas (2026)**†, *Uncertainty-aware AI: Conformal prediction versus reinforcement learning for optimal trade execution*, Statistics, Optimization & Information Computing 16(2), 1334–1349.
- Q: Is a conformal-gated forecast policy more reliable than off-the-shelf RL for VWAP-benchmarked execution?
- M: A gradient-boosted return forecaster with a split-conformal interval. The policy acts on the forecast only when |forecast| > κ·(interval half-width); otherwise it tracks VWAP.
- D: A simulator with stochastic volatility and AR(1) momentum; an empirical coverage check on 30 US large caps.
- E: The gate produces a tunable cost-variance frontier and is more reproducible than PPO (3 seeds).
- L: The gate guards a *forecast*, not an RL policy. Its uncertainty measures forecast dispersion, not the policy's regret. There is no distribution-shift suite and no matched-budget random control.
- → **The closest prior work in execution.** It establishes that "act only when uncertainty is low, else fall back to a schedule" is useful. Our project differs in what is gated (an RL policy), in what is predicted (counterfactual regret against the fallback rather than interval width), and in how it is evaluated (pre-registered, LOFO, matched controls, ≥10 replicates).

## 3. Robust and adversarial RL for trading

**Spooner & Savani (2020)**, *Robust market making via adversarial reinforcement learning*, IJCAI.
- Q: Can an adversary that perturbs market dynamics make market makers robust to misspecification?
- M: A zero-sum game; the adversary controls drift and execution-model parameters.
- D: A stochastic market-making model in the Avellaneda-Stoikov tradition.
- E: Adversarially trained agents are more robust to parameter shifts and accumulate less inventory.
- L: Market making, not execution. Robustness is trained into a single policy.
- → An alternative to selective intervention. Adversarial training changes the policy everywhere; a gate keeps the specialist policy and switches only when needed.

**Pinto et al. (2017)**, *Robust adversarial RL*, ICML.
**Nilim & El Ghaoui (2005)**, Operations Research; **Iyengar (2005)**, Math. of OR (robust MDPs).
**Tobin et al. (2017)**, *Domain randomization*, IROS.
- → General robust-RL tools. Our RL is deliberately trained in-distribution only, which is the realistic case of a policy fitted to recent history. Robust training would be a complementary treatment.

*Robust RL in finance: modelling market impact with elliptic uncertainty sets* (arXiv:2510.19950, 2025)†.
- Robust RL with impact-model uncertainty sets.
- → Directly relevant to our IMPACT family; it is a robustify-the-policy approach rather than a detect-and-fall-back approach.

## 4. Uncertainty, OOD detection and selective prediction

**Chow (1970)**, IEEE Trans. Information Theory.
**El-Yaniv & Wiener (2010)**, JMLR.
**Geifman & El-Yaniv (2017**, NeurIPS; **2019**, ICML, SelectiveNet).
- Q: When should a predictor abstain?
- M: Reject options; risk-coverage trade-offs.
- E: Risk-coverage curves; AURC.
- → We borrow risk-coverage analysis. Our "abstention" means deferring to TWAP, and our risk is *execution regret*, not classification error.

**Madras, Pitassi & Zemel (2018)**, *Predict responsibly: learning to defer*, NeurIPS.
**Mozannar & Sontag (2020)**, ICML.
- Q: Learn when to defer to an expert by modelling the expert's own errors.
- → RCG is "learning to defer" for sequential control. It is trained on *counterfactual* policy-vs-fallback regret, which a simulator with common random numbers makes observable.

**Hendrycks & Gimpel (2017)**, ICLR; **Lee et al. (2018)**, *Mahalanobis OOD detector*, NeurIPS.
**Lakshminarayanan et al. (2017)**, *Deep ensembles*, NeurIPS; **Ovadia et al. (2019)**, NeurIPS.
- Q: Detect OOD inputs and quantify predictive uncertainty.
- E: OOD AUROC; calibration under dataset shift. Ovadia et al. show that uncertainty quality degrades under shift.
- → Our A_OOD (Mahalanobis) and A_ENS (ensemble disagreement) controls. Our H3 asks the question these papers do not: *does the score predict decision regret?*

**Vovk, Gammerman & Shafer (2005)**; **Angelopoulos & Bates (2023)**, FnT ML.
**Bates et al. (2021)**, *Risk-controlling prediction sets*, JACM.
**Angelopoulos et al. (2024)**, *Conformal risk control*, ICLR.
**Gibbs & Candès (2021)**, *Adaptive conformal inference under distribution shift*, NeurIPS.
- Distribution-free guarantees, all requiring exchangeability or a model of the shift.
- → We did not adopt a conformal threshold for the confirmatory gate. Our test conditions are deliberately *not* exchangeable with the calibration data (LOFO), so marginal guarantees would not hold. The Bayes rule (predicted regret > 0) needs no threshold to tune.

**RL-specific OOD:**
- **Sedlmeier et al. (2020)**†, UBOOD, ICAART: bootstrap-ensemble epistemic uncertainty separates ID from OOD states.
- **Mohammed & Valdenegro-Toro (2021)**, arXiv:2112.02694: a benchmark for OOD detection in deep RL.
- **Nasvytis et al. (2024)**†, *Rethinking OOD detection for RL*, AAMAS: they argue that most RL-OOD benchmarks are artificial.
- **Lütjens, Everett & How (2019)**, ICRA, and **Kahn et al. (2017)**: uncertainty-aware cautious control.
- → Each of these evaluates *detection* (ID vs OOD) or *safety* (collisions), not *regret relative to a fallback*. Following Nasvytis et al., our shift suite is anchored to documented market episodes rather than synthetic noise.

## 5. Safe RL, shielding, fallback and runtime assurance

**Sha (2001)**, *Using simplicity to control complexity* (Simplex architecture), IEEE Software.
**Phan et al. (2020)**†, *Neural Simplex Architecture*, NASA Formal Methods.
- Q: How to run a high-performance learned controller safely.
- M: A decision module switches to a verified baseline controller when the state approaches an unsafe region, and supports reverse switching.
- E: Safety guarantees on control benchmarks.
- L: The switching criterion is *safety-set membership*, defined by a known model, not performance regret.
- → The architectural template: learned controller + baseline controller + decision module. Our decision module is *learned from counterfactual regret* and targets *cost*, not hard safety.

**Alshiekh et al. (2018)**, *Safe RL via shielding*, AAAI.
**Thananjeyan et al. (2021)**†, *Recovery RL*, IEEE RA-L.
- Recovery RL switches to a recovery policy when a learned safety critic predicts constraint violation.
- → Recovery RL is the closest *mechanism*: it switches on a learned risk critic. It targets constraint violation in robotics, is trained on the agent's own data, and does not study shift.

**Rotman, Schapira & Tamar (2020)**†, *Online safety assurance for deep RL*, HotNets / NeurIPS Deep RL workshop, arXiv:2010.03625.
- Q: How to detect online that an RL system (adaptive video streaming) is acting outside its training regime, so that it defaults to a heuristic.
- M: Three uncertainty signals: state novelty, value uncertainty and action uncertainty.
- E: Novelty and value uncertainty work; action uncertainty does much worse; defaulting preserves most of the performance.
- L: Networking domain; preliminary; it tests whether the signals flag OOD, not whether they predict regret.
- → **The closest prior work on the general idea.** We cannot claim novelty for "fall back to a heuristic when RL is uncertain". Their finding that action uncertainty is a weak signal is one we can re-test in execution (A_ENS).

**Thomas, Theocharous & Ghavamzadeh (2015)**, *High-confidence policy improvement*, ICML.
**Laroche, Trichelair & Tachet des Combes (2019)**, SPIBB, ICML.
- Deploy a new policy only if it provably improves on the baseline, with high confidence.
- → The offline analogue of selective trust. They guarantee improvement *in distribution*; our gate addresses deployment *under shift*.

**Altman (1999)**, *Constrained MDPs*; **Achiam et al. (2017)**, CPO, ICML; **García & Fernández (2015)**, JMLR survey.
- Constraint-based safe RL.
- → Out of scope. We have no hard constraint; the objective is expected cost.

**Lykouris & Vassilvitskii (2018/2021)**, JACM; **Purohit, Svitkina & Kumar (2018)**, NeurIPS (algorithms with predictions).
**Rice (1976)**; **Kerschke et al. (2019)**, Evolutionary Computation (per-instance algorithm selection).
- A robustness-consistency trade-off: trust ML advice when it is good, retain worst-case guarantees when it is not. Algorithm selection picks an algorithm per instance from instance features.
- → RCG is per-instance algorithm selection between {RL, TWAP}. TWAP plays the role of the worst-case-robust algorithm.

## 6. Regimes, non-stationarity and distribution shift in markets

**Hamilton (1989)**, Econometrica; **Ang & Timmermann (2012)**, Annual Review of Financial Economics.
- Markets switch between persistent volatility/return regimes.
- → Motivates the VOL and LIQ families. We do **not** claim to detect regimes: our shifts are parametric perturbations of a simulator.

**McLean & Pontiff (2016)**, J. Finance.
- Anomaly returns decline about 58% after publication.
- → Motivates the ALPHA family (signal decay).

**Moallemi & Sağlam (2013)**, *The cost of latency*, Operations Research.
**Hasbrouck & Saar (2013)**, J. Financial Markets.
- Quantify the cost of latency.
- → Our LATENCY family and the latency robustness ranges.

**Cont, Kukanov & Stoikov (2014)**, J. Financial Econometrics; **Kolm, Turiel & Westray (2023)**, Mathematical Finance.
- Order-flow imbalance explains contemporaneous price changes; its out-of-sample predictive R² at short horizons is low single-digit percent.
- → Calibration target for our signal (R² ≈ 1.3% at 30 s).

**Cont (2001)**, Quantitative Finance; **Bouchaud, Bonart, Donier & Gould (2018)**, *Trades, Quotes and Prices*, CUP.
**Vyetrenko et al. (2020)**†, *Get real: realism metrics for robust LOB market simulations*, ICAIF.
- Stylised facts and simulator-realism metrics.
- → Our simulator-validation suite.

**Shimodaira (2000)** (covariate shift); **Koh et al. (2021)** (WILDS, ICML); **Padakandla (2021)**, ACM CSUR (RL in non-stationary environments).
- → The general shift taxonomy. Our families include covariate shifts (VOL, LIQ, SIZE: visible in observations) and concept shifts (ALPHA, IMPACT, LATENCY: same observations, different consequences). The concept shifts are where OOD detectors are expected to fail.

## 7. Statistical methodology

- **Owen (2007)**, *The pigeonhole bootstrap*, Annals of Applied Statistics: a crossed (two-way) bootstrap for crossed random effects. We use it because RL replicates and market episodes are crossed.
- **Holm (1979)**, Scandinavian Journal of Statistics: step-down multiple-comparison correction.
- **Henderson et al. (2018)**, *Deep RL that matters*, AAAI; **Agarwal et al. (2021)**, *Deep RL at the edge of the statistical precipice*, NeurIPS.
  - Seed variance in deep RL is large; few-seed point estimates mislead; report intervals over runs.
  - → Our minimum of 10 replicates and replicate-level resampling.

---

## Synthesis: what is and is not already established

| Claim | Status in the literature |
|---|---|
| RL can beat TWAP/AC in-distribution using market-state features | Established (Nevmyvaka 2006; Ning 2021; Lin & Beling 2020; Fang 2021). Few studies use ≥10 seeds. |
| RL execution policies overfit or degrade off-distribution | Argued and analysed (Zhang 2023; Hambly 2023). Rarely quantified under a pre-specified, literature-anchored shift suite. |
| Falling back to a heuristic when an RL system is uncertain is useful | Proposed and preliminarily supported in networking (Rotman 2020), robotics and control (Simplex/NSA, Recovery RL), and for forecast-gated execution (Irshad & Biswas 2026). **Not novel as a general idea.** |
| OOD/uncertainty scores flag shifted inputs | Established, with known failure modes (Ovadia 2019; Nasvytis 2024). |
| Uncertainty scores predict the *counterfactual regret* of trusting RL against a classical fallback in execution | **Not found.** Prior work evaluates OOD detection, forecast intervals, or safety violations, not decision regret against the fallback. |
| A regret-calibrated gate beats rule, blend and matched-budget random-switching controls on a pre-registered endpoint, on *held-out shift families* | **Not found.** |
| Shifts that leave observations unchanged (concept shift: alpha decay, hidden impact) defeat OOD-based gates in execution | **Not found** as a measured result in execution. |

The novelty claim we make, and its limits, is in [`NOVELTY_STATEMENT.md`](../NOVELTY_STATEMENT.md).
