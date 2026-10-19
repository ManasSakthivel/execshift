# Reproducibility

## Quick reference

| Command | What it does | Runtime (Apple M4 Pro, 14 cores) | Touches the test split? |
|---|---|---|---|
| `make test` | 261 tests: unit, integration, CLI, integrity, regression (80 of them one static scan per module) | 20-60 s | no (reads committed results) |
| `make reproduce-paper-fast` / `execshift reproduce --fast` | every pipeline stage at toy scale; writes only `results/fast/`, `models_fast/` | ~2 min | **no** |
| `make reproduce-paper` / `execshift reproduce --full --yes` | the whole paper from scratch, then the verifier and the tests | 76.5 min wall, ~593 CPU-min | yes: appends 780 logged accesses |
| `execshift reproduce --verify [--strict]` | compares `results/authoritative.json` with the committed reference | < 1 s | no |
| `execshift report --study` | re-renders study documents and figures from committed artifacts | ~10 s | no |
| `execshift evaluate --experiment <id>` | re-executes a recorded post-study run, reports identical / different numbers | seconds | no |

Both reproduction targets keep the semantics they had at the confirmatory run. The refactor
moved the stage code from `scripts/` into `adexec/pipeline/` and left every output byte-identical:
`results/`, `docs/FINAL_RESEARCH_REPORT.md`, the tables and the figures regenerate exactly,
and `tests/test_reproducibility.py` pins their SHA-256.

## 1. Environment

- Python **3.12** (recorded: 3.12.14). The lock pins 3.12-only wheels, so 3.11 is not supported; byte-identity is claimed only for
  the recorded environment.
- `make install` builds `.venv` with `uv` if present, otherwise with `python3.12 -m venv`, from
  `requirements-lock.txt`, then installs the package editable (`pip install -e . --no-deps`),
  which also provides the `execshift` / `quant-exec` commands.
- No network access is needed after installation, and there is no external data.

## 2. Dependencies

`requirements-lock.txt` pins every runtime package (numpy 2.5.3, scipy 1.18.1, pandas 3.0.6,
scikit-learn 1.9.1, torch 2.14.1, stable-baselines3 2.9.0, gymnasium 1.3.0, matplotlib
3.11.2, ...). CI installs the CPU wheel of torch 2.14.1. `pyproject.toml` declares minimum
versions for library use. The lock is the reproduction contract. Development tools (pytest,
ruff) are also pinned there for historical reasons (audit item D14). mypy and pre-commit are
optional extras (`pip install -e .[dev]`).

## 3. Seeds

| Randomness | Seed | Where |
|---|---|---|
| market episodes (study) | `SeedSequence([split_tag, condition_index, episode])`, tags 11/22/33/44/55 | `sim.episode_rngs`, `guard.split_rngs` |
| market episodes (post-study benchmarks) | `SeedSequence([66, condition_index, seed, episode])` | `market.benchmark_rngs` |
| PPO member (study) | `seed = 5·replicate + member`; RL-training stream `impact_index·1000 + seed` | `rl.member_seed`, `rl.train_member` |
| PPO member (post-study) | config `[training] seeds`; streams `50,000 + seed` | `agents.training` |
| confirmatory bootstrap | `BOOT_SEED = 20261001`, B = 10,000 | `stats.crossed_bootstrap` |
| H3 bootstrap | seed 7, B = 1,000 | `analysis.h3` |
| figure-data bootstraps | per-condition CIs: `BOOT_SEED`, B = 2,000; control-variate audit: seed 11, B = 1,000 | `pipeline.figure_data` |
| benchmark comparisons | seed 0, B = 2,000 (configurable) | `evaluation.statistics.paired_comparison` |
| RCG gradient boosting | `random_state=0` | `gates.RCG_MODELS` |

No component reads the global NumPy or Python RNG.

## 4. Deterministic settings

- `torch.set_num_threads(1)` in every PPO training and evaluation worker. Required for
  bit-for-bit identical weights. `tests/test_reproducibility.py` trains the same member
  twice and compares weight hashes.
- PPO on CPU only (`device="cpu"`).
- Process pools return results in submission order (`executor.map`), so worker count does
  not change any output.
- Gate thresholds, regressors and blend weights are fitted deterministically from the data.

## 5. Expected runtime

See the quick reference. The full pipeline is dominated by PPO training (150 members ×
20 M steps). On a 4-core machine expect 4-5 h. Post-study benchmark suites take seconds with
existing checkpoints (`shift`: ~5 s; `gate`: ~13 s; `robustness`: ~7 s on one core).

## 6. Expected artifacts

| Artifact | Committed? | Produced by |
|---|---|---|
| `results/authoritative.json` (+ `.reference.json`) | yes, hash-pinned | stage 5 |
| `results/figure_data.json`, `results/tables.md` | yes, hash-pinned | stages 6, 7 |
| `results/simulator_validation.json`, `results/baselines_*.json` | yes, hash-pinned | stages 1, 2 |
| `results/rl_manifest.json` | yes | stage 3 (weight hashes) |
| `results/test_access_log.jsonl` | yes, append-only | every test-split access |
| `docs/FINAL_RESEARCH_REPORT.md`, `FINAL_STANFORD_READINESS_REPORT.md`, `docs/SIMULATOR_VALIDATION.md`, `docs/BASELINES.md` | yes | stage 7 |
| `figures/fig1..fig10` | yes | stages 1, 6 |
| `results/post_study/supplementary_data.json`, `figures/supplementary/` | yes (POST-STUDY DESCRIPTIVE) | `make supplementary` |
| `models/` (22 MB), `results/raw/` (343 MB) | no (git-ignored) | stages 3-5 |
| `runs/<id>/` | no (git-ignored) | `execshift benchmark / evaluate / train` |

## 7. Known nondeterministic components

- **Checkpoint zip files** contain a save timestamp. Weights are fingerprinted instead
  (`policy.pth` SHA-256).
- **Cross-platform floating point.** Different BLAS or torch builds can change PPO weights
  slightly. The verifier's pre-declared tolerance, max(0.02 bp, 10% of the CI half-width),
  plus identical decisions, is the cross-platform criterion. It has not yet been exercised on
  a second platform.
- **PNG figures** can differ byte-wise across platforms (font rasterisation). CI therefore
  checks generated *documents*, not figures, for staleness.
- **Run ids and manifests** contain UTC timestamps and the git state. Two runs of the same
  config have different ids and the same `config_hash`, and their numbers are identical.
- **The test-access log** grows by 780 lines per full reproduction. This is by design.

## 8. Hardware assumptions

CPU only; no GPU is used or needed. The recorded run used an Apple M4 Pro (14 cores, 48 GB
RAM) on macOS. A full reproduction needs about 4 GB of RAM per evaluation worker at peak and
~0.5 GB of disk for checkpoints and raw outputs. The fast pipeline and the test suite run
on any machine that installs the lock, including the 2-core GitHub runners.

## 9. CI tiers

| Trigger | Workflow | What runs |
|---|---|---|
| pull request, push to `main` | `.github/workflows/ci.yml` | ruff, mypy, pytest on 3.12, lock check, strict verify of committed results, generated-document staleness, CLI smoke |
| nightly, manual | `nightly.yml` | `execshift reproduce --fast` |
| release, manual | `reproduce.yml` | `make reproduce-paper` (long; a large or self-hosted runner is recommended) |

The repository is at <https://github.com/ManasSakthivel/execshift>. CI (`ci.yml`) passes on
GitHub for every job. `nightly.yml` and `reproduce.yml` have not yet run there.

## 10. Historical record: clean-clone reproduction (performed 2026-10-02)

**Procedure.** `git clone` of `main` into an empty directory, then `make reproduce-paper-fast`
followed by `make reproduce-paper WORKERS=13`. Nothing was copied in from the original
working tree: environment, data, models and results were all regenerated.

| Stage | Outcome |
|---|---|
| Environment | Built from `requirements-lock.txt` (Python 3.12.14, torch 2.14.1, numpy 2.5.3, SB3 2.9.0, scikit-learn 1.9.1) |
| Data | Regenerated from seeds; no external data |
| Simulator validation | All checks PASS; `results/simulator_validation.json` byte-identical |
| Baseline tuning | Identical tuned parameters; `results/baselines_*.json` byte-identical |
| PPO training | **150/150 members have identical network weights** (`policy.pth` SHA-256 in `results/rl_manifest.json`) |
| Validation and confirmatory evaluation | `results/authoritative.json` **byte-identical** to the committed reference |
| Statistics, figures, tables, report | `results/figure_data.json`, `results/tables.md` and `docs/FINAL_RESEARCH_REPORT.md` byte-identical |
| `scripts/verify_reproduction.py` | 1,063 values and decisions checked; 0 outside tolerance |
| Test suite in the clone | all pass |
| `make reproduce-paper-fast` | PASS in 1 min 51 s; output identical to the original run |

**Runtime.** 76.5 min wall-clock for the full pipeline: 593 CPU-minutes of user time.

**Hardware.** Apple M4 Pro (14 cores, 48 GB RAM), macOS, CPU only.

### Tolerances and nondeterminism (as recorded)

- **Same platform:** the pipeline is bit-for-bit deterministic. Single-threaded torch, fixed
  seeds, `SeedSequence`-derived market streams and a seeded bootstrap.
- **Checkpoint files:** they are *not* byte-identical across runs, because Stable-Baselines3
  writes a save timestamp into the zip. We therefore fingerprint the weights (`policy.pth`)
  instead of the file.
- **Other platforms** (e.g. Linux x86-64, or different BLAS / torch builds): floating-point
  differences in PPO training can change the trained weights slightly.
  `verify_reproduction.py` accepts any point estimate or CI bound within
  max(0.02 bp, 10% of the CI half-width), and requires every hypothesis decision to be
  identical. Cross-platform full reproduction was **not** performed. CI runs only the
  reduced pipeline.
- **Test-access log:** a reproduction appends 780 entries to
  `results/test_access_log.jsonl` (one per replicate × condition × configuration). This is
  expected, and is the audit trail working as designed.

Since that record, one sentence of generated prose in `docs/FINAL_RESEARCH_REPORT.md` has
been corrected ("never acts" became "almost never acts (at most 0.6% of orders)" for the
transient-impact gate). No number or result file changed.

## 11. Fresh-machine walkthrough (verified 2026-10-19)

These commands were run in a new clone of `main` (commit `801a170`) on macOS arm64. The
timings are from that run.

```bash
git clone https://github.com/ManasSakthivel/execshift.git && cd execshift
make install                              # Python 3.12 venv + locked dependencies
make check                                # 258 passed (pytest 40 s; 57 s with ruff and first imports)
make reproduce-paper-fast                 # status PASS in 66 s; writes results/fast/, models_fast/
.venv/bin/execshift inspect --lock        # "pre-registration lock OK ... 10 frozen files"
.venv/bin/execshift reproduce --verify --strict   # 1063 checked, 0 outside tolerance; 12498/12498 identical
```

`make install` needs Python 3.12. With `uv` installed, uv provides it. Otherwise
`python3.12` must be on the `PATH`. Installation time depends on the package cache: it took
2 s with a warm uv cache, and downloading torch takes longer on a cold one. Afterwards the
working tree was unchanged (`git status` clean). Two runs of `make reproduce-paper-fast`
printed identical numbers.

The full study (`make reproduce-paper`) was not re-run on 2026-10-19. Its last full run is
the 2026-10-02 record in §10.

## 12. Deterministic reproduction vs approximate replication

| Claim | Holds | Evidence |
|---|---|---|
| Committed results are internally consistent and match the reference copy | everywhere | `reproduce --verify --strict`, pinned hashes in `make check` |
| Derived documents and tables regenerate exactly from the committed results | everywhere | CI `execshift report --study && git diff --exit-code` |
| A full re-run is **byte-identical** | the recorded environment only (macOS arm64, Python 3.12.14, the lock) | §10 |
| A full re-run on another platform matches **within tolerance**, with identical decisions | expected, **not yet demonstrated** | verifier tolerance max(0.02 bp, 10% of the CI half-width) |
| Simulator episodes, baseline tuning and statistics are seed-deterministic | wherever NumPy's `SeedSequence` streams match (any platform with the locked NumPy) | `tests/test_reproducibility.py` regression pins |

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for contourpy==1.4.0` | Python 3.11 or older | use Python 3.12 (`uv venv --python 3.12`) |
| `make install` cannot find `python3.12` | no `uv` and no `python3.12` | install `uv`, or Python 3.12 |
| `ContaminationError: ... changed after the pre-registration lock` | a frozen file was edited or reformatted | `git checkout -- <file>` and never reformat frozen files |
| `test_historical_artifacts_are_unchanged` fails | a result file was overwritten, for example by a full re-run on another platform | `git checkout -- results/`, then compare with `execshift reproduce --verify` |
| `python -m adexec.reporting.supplementary --recompute` prints "S4-S7 skipped" | `results/raw/` is absent (it is git-ignored) | plain `make supplementary` draws S4-S7 from the committed `results/post_study/supplementary_data.json`. Recomputing that file needs the raw outputs from `make reproduce-paper` |
| benchmark output lists PPO policies under `skipped` | no checkpoints in `models/` | `make train` (or `execshift train --study`), or use `configs/examples/evaluate_liquidity.toml`, which needs no PPO |
| `make typecheck` fails with `No module named mypy` | mypy is optional | `.venv/bin/pip install mypy` |
| PNG figures differ byte-wise after regeneration | font rasterisation differs by platform | expected. Documents, not figures, are checked for staleness |
