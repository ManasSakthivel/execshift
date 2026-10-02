# Novelty statement

> **Question:** What does this project add that existing work does not already establish?

The evidence behind this statement is [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md),
including a second, adversarial search pass (section 8). The verdicts below were written
**before** the confirmatory results were inspected. We make no claim on the grounds that "no
identical implementation was found". Each verdict names the closest prior work.

## Verdicts

| ID | Contribution | Verdict |
|---|---|---|
| C1 | Regret-centred formulation: predict the counterfactual regret of RL versus TWAP; gate = fall back iff predicted regret > 0 | **PLAUSIBLY NOVEL**, and **only** as an *empirical question in execution*. The gating *method* is ALREADY ESTABLISHED. |
| C2 | Control-variate (martingale) regret labels and estimator | **METHODOLOGICAL RIGOR** |
| C3 | Pre-registered leave-one-shift-family-out (LOFO) benchmark | **METHODOLOGICAL RIGOR** (the shift suite itself is an engineering contribution) |
| C4 | Matched-budget random switching, rule and blend controls; ≥10 replicates; crossed bootstrap | **METHODOLOGICAL RIGOR** |
| C5 | The empirical findings | **PLAUSIBLY NOVEL**, contingent on the confirmatory evidence (see below) |

**The central contribution is not a new gating method.**
- Falling back from an uncertain RL policy to a conservative policy is established:
  - Rotman, Schapira & Tamar 2020;
  - Neggatu, Houssineau & Montana, AAMAS 2025;
  - Simplex and Neural Simplex; Recovery RL;
  - in trading, FineFT (Qin et al. 2025);
  - in forecast-gated execution, Irshad & Biswas 2026.
- "Defer iff the expected cost of the model exceeds the expert's" is cost-sensitive / post-hoc learning to defer (Mozannar & Sontag 2020; Narasimhan et al. 2022).
- RCG is an instance of these.

**The strongest remaining research question**, and the one this project is designed to answer, is empirical:

> In RL trade execution, is the regret of trusting the RL policy relative to TWAP
> predictable from pre-trade observables and policy uncertainty, under held-out kinds of
> natural market shift? Does predictability differ between covariate shifts (visible in
> observations) and concept shifts (invisible)?

## Per-claim detail

### C1: PLAUSIBLY NOVEL (empirical question only)

**Closest prior work**
- Neggatu et al. (AAMAS 2025): an uncertainty-thresholded RL / behaviour-cloning switch.
- FineFT (2025): VAE-OOD routing to a conservative trading policy.
- Rotman et al. (2020): uncertainty-triggered defaulting.
- Schneider et al. (2026): evaluate uncertainty by decision utility.
- Mozannar & Sontag (2020) and Narasimhan et al. (2022): cost-sensitive deferral.

**Precise difference**
- Prior RL and trading gates threshold an *uncertainty or OOD* score. They report the *aggregate* performance of the switched system.
- They do not test, episode by episode, whether the score ranks the *counterfactual regret against the specific fallback*. Common random numbers make that regret measurable here.
- They do not test it on *held-out kinds* of shift, against a random control with the *same fallback budget*.

**Why it matters**
- A score that rises under shift but does not track regret leads to harmful fallbacks: it switches off RL exactly where RL is still better.
- Aggregate gains can come from the fallback rate alone, not from *which* episodes were switched. The matched random control (A10) isolates that.

**Evidence required**
- H3: rank correlation and AUROC of the score against regret, with CIs excluding chance on the pooled test.
- H4/H5: the gate must beat always-RL and the matched random control. Otherwise the score's ranking has no decision value.
- If H3/H4 fail, C1 survives only as a **negative empirical answer** to the question above. That is still informative, but it is not a method contribution.

### C2: METHODOLOGICAL RIGOR

- **Closest prior work:** Henderson & Glynn (2002), martingale control variates; their standard use in Monte Carlo finance.
- **What we do:** apply the same device to execution P&L. We subtract `Σ_t q_{t+1}·(innovation_{t+1})` and the latency-bridge term. Both are products of decision-time-measurable inventory and post-decision innovations, so each term has conditional mean zero.
- **What is ours:** using the result as the gate's training label and as the estimator for paired comparisons. The variance reduction is **measured** on the final data and reported with uncertainty in `docs/FINAL_RESEARCH_REPORT.md`. We claim only the reduction we observe.

### C3: METHODOLOGICAL RIGOR

- **Closest prior work:** leave-one-domain-out evaluation (Gulrajani & Lopez-Paz 2021, DomainBed).
- **What we do:** apply it to six execution shift families, which were frozen before training.
- **Value:** it prevents a gate from being credited for detecting shift types it was fitted on. This is rigor, not a new idea.

### C4: METHODOLOGICAL RIGOR

- Matched random controls, ≥10 replicates (Agarwal et al. 2021; Henderson et al. 2018), crossed bootstrap (Owen 2007) and Holm correction are good practice.
- They distinguish this study from the 1-5-seed execution-RL studies we reviewed. They are not a contribution to knowledge in themselves.

### C5: PLAUSIBLY NOVEL (contingent)

- **What it would mean:** a quantified, pre-registered answer to the question above in a validated synthetic market. That covers the size and robustness of RL's in-distribution edge over *tuned* baselines (including a feature-driven heuristic), which natural shifts erode it, and whether uncertainty and OOD signals predict the resulting regret.
- **Closest prior work:** Zhang et al. (2023) on generalisation; FineFT on OOD routing.
- **Difference:** neither reports whether uncertainty predicts regret against the fallback, nor separates covariate from concept shift.
- **Evidence required:** the confirmatory results themselves. Every finding claimed in the report must be backed by a pre-registered or explicitly exploratory analysis, and labelled as such.

## Scope limits

- Every result is from a **synthetic** simulator (validated against stylised facts) and holds conditional on its impact models. Nothing here is evidence about live markets.
- The contribution is **an empirical answer, plus a rigorous benchmark**, to a narrow, well-posed question. It is not a new algorithm.
