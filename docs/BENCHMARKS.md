# Benchmarks

> **Every number on this page is POST-STUDY / EXPLORATORY.** The benchmarks draw new
> episodes from a stream the pre-registered study never used (split tag 66). They are not the
> confirmatory test set, and they neither replace nor extend the study's verdicts
> ([`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md)). Where they disagree with the
> study, the study's design is stronger: 10 replicates, a crossed bootstrap and a locked
> protocol, against one replicate here.

## 1. The framework

A benchmark is declarative data (`adexec/benchmarks/suite.py`):

| Field | Meaning |
|---|---|
| `policies` | `{name: spec}`, built by `agents.make_policy`, rebuilt per impact model |
| `scenarios` | `Scenario(condition, impact_model, latency_s, overrides)` |
| `seeds`, `n_episodes` | episodes per scenario = n_episodes × len(seeds), drawn from `SeedSequence([66, condition, seed, i])` |
| `metrics` | names from `evaluation.metrics.METRICS`, computed on realised IS |
| `reference` | every other policy is compared with it, paired by episode |
| `statistics` | paired percentile bootstrap on control-variate IS, B = 2,000, seed 0; practical threshold 0.25 bp (pre-registered SESOI) |

The runner (`benchmarks/runner.py`) contains no policy names. Every policy in a scenario sees
the same episodes, and every impact-model or latency variant of a condition sees the same
exogenous randomness. A policy whose checkpoints are missing is skipped and recorded. The
reference policy is never skipped.

## 2. Standard suites

| Suite | Category | Scenarios | Episodes | Needs PPO checkpoints |
|---|---|---|---|---|
| `smoke` | smoke | ID, LIQ-sev | 200 | no (CI) |
| `classical` | baseline | 13 conditions | 1,000 | no |
| `baseline` | baseline | ID | 4,000 | yes |
| `seeds` | in-distribution | ID, all 10 replicates | 2,000 | yes |
| `shift` | distribution shift | ID + the 12 frozen shifts, unchanged | 1,000 | yes |
| `impact` | impact robustness | 5 conditions × 3 impact models | 1,000 | yes |
| `latency` | latency robustness | ID at 0, 0.2, 1, 2, 5, 10 s | 1,000 | yes |
| `liquidity` | liquidity robustness | ID, LIQ-mod, LIQ-sev, **LIQ-extreme (NEW)** | 1,000 | yes |
| `risk` | risk | 13 conditions; std, p95, CVaR95, CVaR99 | 2,000 | yes |
| `gate` | gate | 13 conditions; OOD / ensemble / shadow gates rolled out | 1,000 | yes |
| `robustness` | composite | 13 conditions × 3 impact models (39) | 500 | yes |

```bash
execshift benchmark --list
execshift benchmark --suite shift                 # ~5 s on one core with the study's checkpoints
execshift benchmark --suite robustness --episodes 200
```

PPO checkpoints are not committed (22 MB, git-ignored). Regenerate them with `make train`;
their weight hashes are in `results/rl_manifest.json`. Suites that need them skip RL cleanly
when they are absent. The suites marked "no" run anywhere.

## 3. Results (run 2026-10-03, study checkpoints, replicate 0)

All runs: macOS arm64, Python 3.12.14, the git commit recorded in each manifest. Rerun
with the command shown. Same-platform reruns reproduce every number exactly
(`execshift evaluate --experiment <id>` reports `0 different`).

Values are `policy − TWAP`, the paired difference in expected IS (control variate), in bp,
with a 95% bootstrap CI over episodes. Negative favours the policy.

### 3.1 Baselines, in-distribution (`baseline`, 4,000 episodes; config hash `3599c8d5`)

| Policy | − TWAP (bp) | win rate | verdict |
|---|---|---|---|
| AC (tuned) | −0.02 [−0.05, +0.02] | 50% | practically equivalent |
| ADAPT (tuned) | −2.29 [−2.41, −2.17] | 76% | practically better |
| RL (5-member ensemble) | −2.52 [−2.63, −2.42] | 87% | practically better |
| RL (single member) | −2.39 [−2.51, −2.28] | 82% | practically better |

Consistent with the study's H1 (−2.45 [−2.57, −2.34]). RL and ADAPT are within 0.25 bp of each
other here as well.

### 3.2 Training-seed variability (`seeds`, 2,000 episodes; `5f214f1c`)

All 10 replicates: RL − TWAP between −2.43 and −2.51 bp, every CI excludes zero, win rates
82-88%. The spread across replicates (0.08 bp) is small relative to the effect.

### 3.3 Distribution shift (`shift`, 1,000 episodes per condition; `313552d2`)

| Condition | RL − TWAP | ADAPT − TWAP | Note |
|---|---|---|---|
| ID | −2.64 [−2.88, −2.41] | −2.30 [−2.53, −2.04] | |
| VOL-mod / VOL-sev | −4.17 / −7.41 | −3.83 / −4.63 | RL strongest under volatility shift |
| LIQ-mod | −1.65 [−1.85, −1.47] | −1.40 | |
| LIQ-sev | **−0.19 [−0.41, +0.03]** | **+3.02 [+2.60, +3.43]** | replicate 0 only; all 10 replicates: +0.95 (see 3.4) |
| ALPHA-mod | −0.98 | −1.00 | |
| ALPHA-sev | **+0.60 [+0.56, +0.65]** | +0.37 | RL loses to TWAP (hidden concept shift) |
| IMPACT-mod / -sev | −1.71 / −1.04 | −1.74 / −1.19 | |
| LAT-mod / -sev | −2.58 / −2.32 | −2.46 / −2.06 | |
| SIZE-mod / -sev | −2.27 / −2.29 | −2.22 / −2.09 | |

### 3.4 LIQ-sev: a single-replicate benchmark can mislead (and did, at first)

The `shift` suite uses RL replicate 0. In LIQ-sev it gives RL − TWAP = −0.19 [−0.41, +0.03] bp,
which is inconclusive. The study found RL clearly worse there: +0.97 [+0.47, +1.43] bp over
10 replicates. The two intervals do not overlap.

The explanation is training-seed variance, not the new episodes. In the study's own test
outputs, replicate 0 is the most favourable of the ten in LIQ-sev (−0.18 bp; the other nine
range from +0.48 to +1.72). Running all ten replicates on the new episodes
(`configs/examples/evaluate_liqsev_replicates.toml`, config hash `143b6ddb`):

| replicate | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RL − TWAP (bp) | −0.19 | +0.51 | +1.48 | +0.47 | +1.70 | +0.72 | +1.25 | +1.37 | +1.16 | +1.07 | **+0.95** |

That reproduces the study (+0.97) on episodes it never saw. Replicate 0 was not selected for
looking good: it is simply the first, fixed before any benchmark ran. **Lesson, now built
into the docs:** single-replicate intervals omit training-seed variance (METHODOLOGY §8). For
any claim about RL under shift, use all replicates. The `seeds` suite and this config show
how. In the NEW harsher point (LIQ-extreme), even replicate 0 is clearly worse than TWAP:
+1.28 [+0.97, +1.61] bp.

### 3.5 Impact-model robustness (`impact`, `robustness`)

Under each impact model, the conditions where RL is practically worse than TWAP
(`robustness`, 500 episodes per scenario; `f8273d6e`):

| Impact model | RL practically worse than TWAP in |
|---|---|
| sqrt (primary) | ALPHA-sev |
| linear | LIQ-mod, LIQ-sev, ALPHA-sev, IMPACT-sev |
| transient | ALPHA-sev |

Under linear impact the LIQ-sev regret is very large: RL − TWAP = +16.96 [+15.75, +18.33] bp
(`impact`). This matches the study's H6 mechanism. The room a fallback gate has to help
depends on the impact model more than on the gate.

### 3.6 Latency robustness (`latency`; `0f35b7ef`)

RL − TWAP in-distribution: −2.65 (0 s), −2.64 (0.2 s, nominal), −2.63 (1 s), −2.61 (2 s),
−2.57 (5 s), −2.48 (10 s). RL's edge shrinks slowly and monotonically with latency and
remains practically significant at 10 s. ADAPT shows the same pattern. Consistent with the
study's H6 latency result (PASS).

### 3.7 Tail risk (`risk`, 2,000 episodes per condition; `55f8d08d`)

CVaR95 of realised IS (bp):

| Condition | TWAP | ADAPT | RL |
|---|---|---|---|
| ID | 75.4 | **60.3** | 69.9 |
| VOL-sev | 227.9 | **166.4** | 213.4 |
| LIQ-sev | **96.9** | 123.1 | 102.4 |
| ALPHA-sev | 70.2 | **66.1** | 76.6 |
| IMPACT-sev | 83.3 | **68.8** | 80.6 |

RL has the lowest *mean* cost in most conditions. ADAPT has the lowest *tail* cost in 11 of
13 conditions. RL's SD exceeds TWAP's everywhere. The study found the same (ID CVaR95: RL
79.1 vs ADAPT 64.9). The objective is risk-neutral, and it shows.

### 3.8 Uncertainty gates rolled out (`gate`, 1,000 episodes per condition; `6f7fbdfa`)

Mixture-weighted (study weights), these are means over 13 conditions, not a confirmatory test:

| Policy | Mixture mean IS (bp) | Mixture fallback rate |
|---|---|---|
| RL (always) | **5.98** | 0% |
| ensemble gate (TRAIN-ID 95% quantile) | 6.14 | 6.2% |
| OOD gate (TRAIN-ID 95% quantile) | 6.56 | 20.2% |
| shadow gate (threshold 0) | 6.92 | 40.1% |
| TWAP (always) | 8.39 | 100% |

**No single-signal gate beats always-RL.** The OOD gate falls back on 97% of VOL-sev orders,
the regime where RL's advantage is largest (−7.41 bp), and gives most of that advantage away
(−0.21 bp). It falls back on only 7% of ALPHA-sev orders, where RL actually loses. This is
the study's finding 3, "OOD is not regret", reproduced on new episodes. The ensemble gate
barely reacts to any shift. Fallback heatmap: `runs/gate-*/figures/gate_intervention_frequency.png`.

## 4. Adding a suite

```python
# adexec/benchmarks/catalog.py
def my_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        "my-suite", "custom", "What question this answers.",
        named_policies([{"kind": "twap"}, {"kind": "adapt", "tuned": True}]),
        (Scenario("ID"), Scenario("ID", overrides={"sigma_mult": 2.0}, label="VOL-2x (NEW)")))
SUITES["my-suite"] = my_suite
```

Name every non-pre-registered scenario `... (NEW)`. `tests/test_benchmarks.py` enforces this.
