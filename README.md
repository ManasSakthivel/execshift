# When should an RL execution policy be trusted?

**A pre-registered, synthetic-market study of regret-calibrated selective intervention for
RL trade execution under distribution shift.**

> **Status: BLOCKED — research not yet ready** (see
> [`FINAL_STANFORD_READINESS_REPORT.md`](FINAL_STANFORD_READINESS_REPORT.md)).
> The study is complete, and its central positive claim was **not supported**. This
> repository reports a rigorous negative/mixed result, not a working method.

All market data here are **synthetic**, produced by a simulator validated against stylised
facts. Nothing in this repository is evidence of real-market performance, profitability or
deployability.

## Question

In this study's synthetic market, an RL execution agent beats tuned classical schedules on
the kind of market it was trained on. Under distribution shift it may not.

Can an uncertainty-aware gate decide, at order arrival, when to **trust RL** and when to
**fall back to TWAP**? Specifically: does uncertainty predict the *counterfactual regret*
`IS_RL − IS_TWAP`, and does acting on it beat simple controls?

## Headline results

Confirmatory test split: 10 RL replicates × 5-member PPO ensembles, 16,000 episodes. All
values are in basis points of implementation shortfall, with 95% crossed-bootstrap CIs
(full tables: [`results/tables.md`](results/tables.md)).

| Hypothesis | Result | Verdict |
|---|---|---|
| H1: RL beats tuned TWAP in-distribution | RL − TWAP = -2.45 [-2.57, -2.34] bp; practically tied with a tuned heuristic (ADAPT) | PASS |
| H2: RL degrades under natural shift | edge erodes in 8 of 12 conditions; RL loses to TWAP only under severe liquidity drought and complete alpha decay | PASS |
| H3: the gate's score predicts regret | Spearman ρ = +0.108, AUROC 0.542; ensemble and OOD uncertainty do not predict regret | PASS (statistically), practically weak |
| H4: the gate beats always-RL | RCG − RL = -0.013 [-0.024, -0.004] bp, below the 0.25 bp pre-registered threshold | **FAIL** |
| H5: the gate beats blend, rule, matched-random and TWAP controls | yes under the primary impact model, but against matched random switching only -0.005 [-0.010, -0.001] bp | PASS (primary model only) |
| H6: robust to impact model and latency | robust to latency; **not** robust to the impact model | **FAIL** |

**What the study establishes**, within its synthetic scope:

1. A fallback gate is worth at most the *headroom* RL leaves: the cost RL incurs in the
   regimes where it loses to TWAP. Here that headroom depends on the impact model more than
   on the gate.
2. Neither policy-ensemble disagreement nor OOD distance predicts execution regret. OOD
   distance flags volatility shifts, which is exactly where RL is strongest.
3. Concept shifts that leave observations unchanged, such as alpha decay, are invisible to
   any pre-trade gate.

Full write-up: [`docs/FINAL_RESEARCH_REPORT.md`](docs/FINAL_RESEARCH_REPORT.md).

## How the study was run

1. **Simulator validated before any policy claim.**
   [`docs/SIMULATOR_VALIDATION.md`](docs/SIMULATOR_VALIDATION.md): stylised-fact checks
   under three impact models. One failure was found and fixed before training.
2. **Literature review and adversarial novelty audit.**
   [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) and
   [`NOVELTY_STATEMENT.md`](NOVELTY_STATEMENT.md). The gating *method* is established; the
   contribution is the empirical answer plus a rigorous benchmark.
3. **Pre-registration hash-locked before the test split existed.**
   [`research/PREREGISTRATION.md`](research/PREREGISTRATION.md),
   [`research/prereg.lock`](research/prereg.lock). Every test access is logged in
   `results/test_access_log.jsonl`.
4. **Fair baselines.** [`docs/BASELINES.md`](docs/BASELINES.md): tuned TWAP, AC and ADAPT,
   with the same in-distribution information as RL.
5. **RL protocol.** [`docs/RL_TRAINING.md`](docs/RL_TRAINING.md): 150 PPO members, a
   16-configuration search on validation data only, and no best-run selection.
6. **Statistics.** Common random numbers, a crossed replicate × episode bootstrap, Holm
   correction, effect sizes and a smallest effect size of interest (SESOI). A
   control-variate estimator is used, with realised-cost sensitivity for every comparison.

## Reproduce

```bash
make reproduce-paper-fast   # reduced end-to-end sanity pipeline (~2 min on 14 cores; never touches test data)
make reproduce-paper        # full paper from a clean clone (~2-4 h on 14 cores), then verifies against
                            # results/authoritative.reference.json within documented tolerances
make test lint              # unit + scientific-integrity tests, ruff
```

A clean-clone full reproduction was byte-identical to the committed results ([`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)). Dependencies are pinned in `requirements-lock.txt` (Python 3.12). Checkpoints are not
committed; their SHA-256 hashes are in `results/rl_manifest.json`.

## Repository map

```
adexec/        simulator, policies, PPO, gate features, gates, statistics, analysis, report renderers
scripts/       pipeline stages (validate → tune → train → val → confirm → figures → report → verify)
research/      pre-registration + lock
results/       authoritative results, tables, baselines, validation, manifests, test-access log
docs/          literature review, simulator validation, baselines, RL training, final report
figures/       programmatically generated figures 1-10
tests/         unit tests and scientific-integrity tests (leakage, seeds, stale numbers, units)
```

## Provenance

This repository began as a clone of
[`Bilal-Aamir-Yousuf/rl-market-maker`](https://github.com/Bilal-Aamir-Yousuf/rl-market-maker),
an RL *market-making* project. That project has **no license**, so **none of its code is
included or redistributed**: this is an independent, new codebase on an orphan branch.
See [`PROVENANCE.md`](PROVENANCE.md).

New code is MIT-licensed ([`LICENSE`](LICENSE)). Citation metadata: [`CITATION.cff`](CITATION.cff).
