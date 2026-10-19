# Experiments

Two kinds of experiment live in this repository, and they are kept apart on purpose.

| | Pre-registered study | Post-study experiments |
|---|---|---|
| Label | PRE-REGISTERED CONFIRMATORY | POST-STUDY / EXPLORATORY |
| Run by | `make reproduce-paper` / `execshift reproduce --full` | `execshift benchmark`, `evaluate`, `train` |
| Episodes | test split (tag 33), touched once after the lock | benchmark stream (tag 66); training streams ≥ 50,000 |
| Configuration | frozen Python (`adexec/config.py`); described as data in `configs/study/preregistered.json` | TOML (`configs/examples/`) |
| Outputs | `results/authoritative.json` and derived files (committed, pinned by hash) | `runs/<id>/` (git-ignored; one manifest per run) |
| Provenance | `execshift inspect --claim H4` / `--study` | `execshift inspect <id>` |

## 1. The pre-registered study

Hypotheses H1-H6, decision rules and SESOI are in
[`research/PREREGISTRATION.md`](../research/PREREGISTRATION.md) §3. The design as data:

```bash
execshift inspect --conditions          # the 13 frozen conditions and their parameters
execshift inspect --study               # reconstructed manifest: lock, frozen hashes, seeds, artifacts
cat configs/study/preregistered.json    # episodes, replicates, PPO settings, bootstrap, mixture
```

| Stage | Command | Output |
|---|---|---|
| 1 simulator validation | `execshift reproduce --stage validate-sim` | `results/simulator_validation.json`, Fig. 1 |
| 2 baseline tuning | `--stage tune` | `results/baselines_<impact>.json` |
| 3 PPO training (150 members) | `--stage train` | `models/`, `results/rl_manifest.json` |
| 4 validation evaluation | `--stage val` | `results/raw/val_*.pkl` |
| 5 confirmatory run | `--stage confirm` | `results/raw/test_*.pkl`, `results/authoritative.json`, test-access log |
| 6 figures | `--stage figures` | `results/figure_data.json`, Figs. 2-10 |
| 7 documents | `--stage report` | final report, tables, readiness report |
| 8 verification | `--stage verify` | pass/fail against `results/authoritative.reference.json` |

### Answering "what generated this number?" for the study

```bash
$ execshift inspect --claim H4
{ "claim": "H4", "verdict": "FAIL",
  "value": {"estimate": -0.0131, "ci_low": -0.0239, "ci_high": -0.0037, ...},
  "source": "results/authoritative.json/primary/H45/H4/comparison",
  "computed_by": "adexec/analysis.py::h45 -> compare(RCG, RL, MIXTURE_WEIGHTS)",
  "decision_rule": "RCG - RL significant and below -0.25 bp (pre-registered SESOI)",
  "data": "TEST split (tag 33), 10 replicates x 4,000 ID + 12 x 1,000 shifted episodes; ...",
  "prereg_sha256": "2abf320f...", "results_commit": "0e7f68c3..." }
```

## 2. Post-study experiments

### Configuration

One TOML file per experiment. Unknown keys are errors. Sections:

| Section | Controls |
|---|---|
| `[experiment]` | `name`, `kind` (`evaluate` / `benchmark` / `train`), `description`, `label` |
| `[market]` | `conditions` (any of the 13), `impact_model`, `latency_s`, named regimes `volatility` / `liquidity` / `alpha` / `order_size`, raw `overrides` (regime knobs only) |
| `[[policies]]` | one table per policy spec: `twap`, `ac`, `adapt`, `ppo`, `blend`, `gated` |
| `[model]` | PPO architecture: `pi_layers`, `vf_layers`, `log_std_init` |
| `[training]` | `total_steps`, `n_envs`, `seeds`, PPO hyper-parameters |
| `[evaluation]` | `n_episodes`, `seeds`, `metrics`, `reference`, `bootstrap_B`, `threshold_bp` + `threshold_source` |
| `[benchmark]` | `suite` (+ optional `n_episodes`, `seeds` overrides) |

The named regimes reuse the frozen shift magnitudes: `volatility = "crisis"` is the VOL-sev
multiplier, `liquidity = "drought"` the LIQ-sev multipliers. Combining them produces
multi-axis scenarios that the study never tested. Such scenarios are marked
`is_preregistered = false`.

The config hash is the SHA-256 of the canonical JSON of the normalised config: defaults
filled in, keys sorted, same section layout as the file. Two files that mean the same thing
get the same hash.

### Running

```bash
execshift evaluate --config configs/examples/evaluate_liquidity.toml   # no PPO needed
execshift evaluate --config configs/examples/evaluate_regimes.toml     # needs models/
execshift train    --config configs/examples/train_small.toml          # minutes on CPU
execshift benchmark --suite robustness                                  # see BENCHMARKS.md
```

### The registry

Each run gets `runs/<name>-<hash8>-<UTC timestamp>/`:

```
manifest.json     experiment_id, label, kind, created_utc, git_commit, git_dirty,
                  config_hash, config, seeds, policies, environment (Python, platform,
                  package versions), impact_model, scenarios, metrics, artifact_paths
                  (SHA-256 of every file), software_version, prereg_lock_sha256
config.toml       the exact file used (evaluate / train)
results.json      per-(scenario, policy) metrics and paired comparisons with verdicts
episodes.npz      per-episode realised and control-variate IS (recompute any metric)
REPORT.md         rendered report, labelled POST-STUDY / EXPLORATORY
figures/          comparison vs reference, metric bars, gate intervention heatmap
models/           (train runs) checkpoints; training.json has the weight hashes
```

```bash
execshift inspect                       # list runs
execshift inspect <id-prefix>           # manifest summary + re-hash every artifact
execshift evaluate --experiment <id>    # re-execute; reports how many numbers are identical
execshift report --experiment <id>      # re-render report and figures from results.json
```

`evaluate --experiment` is the reproducibility check for a single run: on the same platform
every number should come back identical (`0 different`).

## 3. Supplementary analyses of the study (POST-STUDY DESCRIPTIVE)

`make supplementary` (or `python -m adexec.reporting.supplementary`) produces
`figures/supplementary/S1`-`S7`. S1-S3 read `results/authoritative.json`. S4-S7 read
`results/post_study/supplementary_data.json`, a small reduction of the git-ignored raw
confirmatory outputs. These analyses run **no new tests** and use **no new data**. The
reduction is checked against the study: for example, the RL CVaR95 in ID equals the value in
`authoritative.json`, and all 160,000 episode-replicates are present in the S6 histogram.

## 4. Rules for new experiments

1. Never request the test split. The static integrity test will fail.
2. Never label a run pre-registered or confirmatory. The config loader refuses it.
3. Any threshold other than 0.25 bp needs a `threshold_source` that says why.
4. Report every run you make, including the inconvenient ones. Runs are cheap, and a
   selectively reported benchmark is worse than none.
5. A post-study result does not change a pre-registered verdict. If it contradicts the
   study, say so, and explain the difference in design: replicates, episodes, scenario.
