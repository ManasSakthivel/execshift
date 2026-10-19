# Executive Assessment

ExecShift is a technically real, reproducible research system built around a pre-registered
study whose central claim failed. The engineering is strong. The frozen study core is wrapped
by typed interfaces, a policy-agnostic evaluation engine, a benchmark framework, a hashed
experiment registry, a CLI, 258 tests and tiered CI. None of the ten hash-locked study files
was modified, and no historical result changed.

The scientific contribution is a well-quantified **negative result**:

- the uncertainty gate's improvement over always-RL is −0.013 bp [−0.024, −0.004], against a
  pre-registered 0.25 bp threshold;
- uncertainty predicts regret only weakly (ρ = 0.108, AUROC 0.542);
- the gate's value depends on the impact model.

The repository reports this result as the finding. It makes no attempt to present a method
that works.

This assessment is evidence-based, not a checklist. Two dimensions are rated PARTIAL. Their
reasons are below, under Reproducibility and Benchmark Quality.

# Technical Depth
**PASS**

Evidence:
- A validated synthetic market: stochastic volatility, Student-t returns, latent alpha,
  spread/depth/volume processes, three impact models and Brownian-bridge latency
  (`adexec/sim.py`). The declared impact exponents are verified empirically: measured
  0.52 / 1.03 / 0.53 vs declared 0.5 / 1 / 0.5 (`tests/test_market.py`).
- 150 PPO members (3 impact models × 10 replicates × 5), a batched SB3 `VecEnv`, and an
  unbiased control-variate reward and estimator (4×-99× variance reduction).
- Statistics match the design: a crossed replicate × episode bootstrap, Holm correction,
  effect sizes, a SESOI, leave-one-family-out gate fitting, and a matched-random control that
  separates *which* orders are gated from *how many*.
- A mechanistic explanation of the failure, measured rather than asserted: fallback headroom
  (0.065 / 0.788 / 0.011 bp), visible vs hidden shift (37% vs 0.1% fallback), OOD ≠ regret.

# Engineering Quality
**PASS**

Evidence:
- The refactor is constrained by a hash lock: the frozen core is unchanged
  (`execshift inspect --lock`), and every platform layer depends on it, never the reverse.
  Adapters are tested to be bit-identical to the frozen code (`tests/test_agents.py`,
  `test_evaluation.py`, `test_uncertainty.py`, `test_metrics.py`).
- `typing.Protocol` interfaces for MarketSimulator, ImpactModel, LatencyModel,
  ExecutionPolicy, RLPolicy, BaselinePolicy, UncertaintyEstimator, EvaluationMetric,
  Benchmark and ArtifactStore. Each has a real implementation and a consumer.
- The evaluation engine and benchmark runner contain no policy names and no `isinstance`
  checks, enforced by a test that inspects their string constants.
- Paths are resolved at call time (`paths.py`), with no import-time global state. The CLI
  calls library functions, not scripts.
- ruff clean. mypy clean on 25 platform modules. Type checking found and fixed a real bug:
  `report --study` called `main` on a function.

# Reproducibility
**PARTIAL**

Evidence:
- Same platform: a clean-clone full reproduction was byte-identical (150/150 identical PPO
  weights, `authoritative.json` byte-identical; 2026-10-02). Today,
  `execshift reproduce --verify --strict` reports 1,063 checked values with 0 outside
  tolerance and 12,498 of 12,498 other numeric leaves identical.
- `make reproduce-paper-fast`: PASS in 65 s (2026-10-03).
- `make reproduce-paper` / `make reproduce-paper-fast` keep their original semantics. The
  regenerated documents, tables and figures are byte-identical after the refactor. SHA-256
  pins on 10 historical artifacts are enforced by tests.
- PPO training determinism is tested (same seed, same weight hash). Post-study runs
  re-execute to identical numbers (`execshift evaluate --experiment <id>` → `0 different`).
- **Why PARTIAL:** cross-platform reproduction has never been run, so the verifier's
  tolerances are untested on a second platform. The full reproduction was not rerun after
  this refactor. Equivalence rests on byte-identical regenerated outputs, the strict verify
  and stage-level tests, not on a second 76-minute run. GitHub Actions has never executed,
  because the repository has not been pushed.

# Benchmark Quality
**PARTIAL**

Evidence:
- A declarative `BenchmarkSuite` covers policies, scenarios, seeds, metrics, reference and
  statistical procedure. There are 11 standard suites covering baseline, in-distribution,
  shift (all 12 frozen conditions, unchanged and tested), impact, latency, liquidity, risk,
  gate and composite robustness (39 scenarios).
- Benchmarks use a separate episode stream (tag 66) that never reaches the test split. Every
  run is labelled POST-STUDY / EXPLORATORY with a full manifest. New scenarios are named
  `(NEW)`.
- Results on real checkpoints are reported in `docs/BENCHMARKS.md`, including an
  uncomfortable one. A single-replicate benchmark made RL look harmless in LIQ-sev
  (−0.19 bp). The 10-replicate run on the same episodes reproduces the study's loss
  (+0.95 vs +0.97 bp). This is documented as a lesson, not hidden.
- **Why PARTIAL:**
  - The standard suites use one RL replicate, so their intervals omit training-seed variance.
    The `seeds` suite and a replicate config exist, but the default suites are not
    multi-replicate.
  - The learned RCG gate cannot be benchmarked outside the confirmatory pipeline.
  - RL suites need git-ignored checkpoints: a 76-minute `make train`.
  - There is no real-data benchmark.

# Testing
**PASS**

Evidence:
- 258 tests, all passing in ~15 s:

  | File | Tests | Covers |
  |---|---|---|
  | `test_integrity.py` | 80 | 80 of them are one static test-split scan per module |
  | `test_experiments.py` | 32 | |
  | `test_reproducibility.py` | 24 | |
  | `test_market.py` | 23 | |
  | `test_benchmarks.py` | 20 | |
  | `test_cli.py` | 17 | |
  | `test_core.py` | 14 | |
  | `test_agents.py` | 12 | |
  | `test_statistics.py` | 12 | |
  | `test_metrics.py` | 10 | |
  | `test_uncertainty.py` | 6 | |
  | `test_evaluation.py` | 5 | |
  | `test_reporting.py` | 3 | |

- Coverage by type:
  - simulator invariants and impact exponents;
  - adapters bit-identical to frozen code;
  - the gated-rollout equals the analytical `where()` used by the study;
  - metrics checked against hand calculations, sklearn/scipy and the frozen code;
  - practical verdicts, including the study's H4 case;
  - config parsing with 11 rejection cases;
  - manifest integrity and tamper detection;
  - CLI commands end to end;
  - regression pins on the engine and SHA-256 pins on historical artifacts;
  - PPO determinism;
  - the verifier catching changed estimates, flipped decisions and truncated lists.
- RL code paths are tested in CI with a tiny trained fixture, not skipped.
- No count padding. The static scan's 80 parametrized cases are disclosed as such.

# Documentation
**PASS**

Evidence:
- `README.md` leads with the problem, why it matters, the architecture, what is implemented,
  the findings, the negative findings, reproduction, benchmarks, limitations and future work.
  It states "The uncertainty gate did NOT achieve the hypothesized practical improvement",
  with H4 −0.013 [−0.024, −0.004] and H3 ρ = 0.108, AUROC 0.542.
- `docs/` contains ARCHITECTURE_AUDIT (before + second-pass forensics), ARCHITECTURE,
  METHODOLOGY (with every metric's unit and interpretation), EXPERIMENTS, BENCHMARKS,
  REPRODUCIBILITY (environment, dependencies, seeds, determinism, runtime, artifacts,
  nondeterminism, hardware, CI tiers), LIMITATIONS and FUTURE_WORK (six directions, each
  with limitation, method, data, design and failure modes). The repository root has
  `CONTRIBUTING.md` and `ATTRIBUTION.md`.
- `execshift inspect --claim H1..H6` traces every headline number to a JSON pointer, its
  generating function, the data split, the decision rule and the results commit.

# Portfolio Value
**PASS**

Evidence:
- It answers "is this merely a trading bot?" with a clear no. It is research infrastructure
  around a falsifiable, pre-registered question, with no profitability claims.
- It shows skills that are hard to fake:
  - refactoring under a cryptographic freeze without touching frozen bytes;
  - designing statistics around crossed random factors;
  - reporting a failed hypothesis precisely;
  - catching a misleading benchmark through replicate analysis.
- Another researcher can extend it. Policies, estimators, metrics, scenarios, suites and
  simulators each plug in at one place (`docs/ARCHITECTURE.md` §10), and the platform will
  label and register the results.
- What it is not: a demonstration of a working method or of real-market skill. A reviewer
  looking for a positive result will not find one, and should not.

# Scientific Integrity
**PASS**

Evidence:
- The pre-registration and lock are unchanged (SHA-256 pinned). All ten frozen modules match
  the lock. `results/` and `research/` are byte-identical to `HEAD`. The test-access log is
  unchanged at 780 entries, all after the lock.
- No threshold, seed, shift definition or hypothesis was altered. The config loader refuses a
  non-0.25 bp threshold without a stated source, and refuses "pre-registered" labels on new
  runs.
- Every new number is labelled:
  - POST-STUDY / EXPLORATORY: benchmarks, on a separate stream;
  - POST-STUDY DESCRIPTIVE: supplementary figures S1-S6, which re-plot frozen outputs
    without new tests and are cross-checked against `authoritative.json`.
- Audit findings are documented, not corrected:
  - S1: stale ADAPT grid text in the pre-registration;
  - S2: the pre-lock timeline is not git-corroborated;
  - S3: verifier coverage.
- The weak H3 relationship is plotted honestly: all 160,000 points, linear axes, no clipping.
  AI-assisted development is disclosed in `ATTRIBUTION.md`.

# Research Limitations

- Synthetic data only; a single parametric market; marketable orders only; no fees.
- Conclusions depend on the impact model (headroom 0.065 / 0.788 / 0.011 bp).
- Gating happens once, at arrival. Mid-order intervention and an ADAPT fallback were not
  studied.
- A risk-neutral objective: RL has higher tail risk than ADAPT (CVaR95 79.1 vs 64.9 bp).
- The bounded action space ([¼, 4] × TWAP pace) was not ablated.
- Full list: `docs/LIMITATIONS.md`.

# Future Research Opportunities

1. Direct regret prediction with failure-mode-specific features. Bounded by headroom, which
   is itself worth demonstrating.
2. Online mid-order intervention, to catch concept shift such as alpha decay.
3. Conformal risk control on regret.
4. Learned, regret-supervised market-regime representations.
5. Cross-market validation, measuring headroom per asset class.
6. Real limit-order-book validation, starting with a pre-registered headroom measurement.

Details: `docs/FUTURE_WORK.md`. None is implemented, to keep the study uncontaminated.

# Remaining Blockers

- **Not pushed; CI has never run on GitHub.** All CI steps pass locally. The first push will
  be the first real CI run.
- **Push hazard.** The local clone still has the unlicensed upstream history on `master` and
  the `upstream` remote. Push `main` only. Never use `git push --all`.
- **Uncommitted work.** The refactor, the new layers, the tests and the docs are uncommitted
  in the working tree. No commit was requested, and no history was rewritten. They should be
  reviewed and committed.
- **Cross-platform reproduction is unverified**, and a post-refactor full 76-minute rerun was
  not performed.
- **The research claim remains BLOCKED** (`FINAL_STANFORD_READINESS_REPORT.md`). Only a new
  pre-registered study can change that.

# Final Recommendation

Ready for a careful GitHub review **after** two owner actions: commit the working tree, and
push `main` only, then let CI run once.

Present the project as rigorous research infrastructure with a pre-registered negative
result: "generic uncertainty does not reliably predict RL execution regret, and fallback
gains are bounded by impact-model-dependent headroom". Do not present it as a working gating
method or as evidence of trading skill.

The engineering, reproducibility and scientific-honesty dimensions are strong. The two
PARTIAL ratings (cross-platform reproduction; single-replicate default benchmarks) are real,
disclosed and fixable without touching the study.
