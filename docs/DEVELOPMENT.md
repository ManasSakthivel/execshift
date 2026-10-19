# Development guide

How to work on ExecShift without damaging the study it contains. The contribution rules and
the pull-request checklist are in [`../CONTRIBUTING.md`](../CONTRIBUTING.md). Read its §1
("What you must not change") first. This page covers the day-to-day workflow.

## 1. The one constraint that shapes everything

Ten files are hash-locked by `research/prereg.lock`:

```text
research/PREREGISTRATION.md
adexec/{config,sim,policies,rl,features,gates,stats,analysis,tuning}.py
```

Changing any byte, including whitespace or an auto-format, makes `guard.check_lock()` refuse
to generate confirmatory test episodes. New behaviour therefore goes into the platform
subpackages (`market/`, `agents/`, `uncertainty/`, `evaluation/`, `benchmarks/`,
`experiments/`, `pipeline/`, `reporting/`, `cli/`), which wrap the frozen core. The package
layout and dependency rules are in [`ARCHITECTURE.md`](ARCHITECTURE.md) §2-3.

## 2. Environment

```bash
make install                            # .venv (Python 3.12) from requirements-lock.txt + editable install
.venv/bin/pip install mypy pre-commit   # optional dev tools (also: pip install -e .[dev])
.venv/bin/pre-commit install            # optional: ruff, lock check, no files > 1 MB on every commit
```

Python 3.12 is required. The lock pins wheels that do not exist for 3.11.

## 3. Conventions actually enforced

| Area | Rule | Enforced by |
|---|---|---|
| Lint | ruff rules E, F, I, UP, B; line length 100 | `make lint`, CI, pre-commit |
| Formatting | `ruff format` is allowed on platform code. It is configured to skip the frozen files. Run no other formatter on them. | `pyproject.toml [tool.ruff.format]` |
| Types | mypy on the platform layers only | `make typecheck`, CI |
| Test-split access | only `pipeline/confirmatory.py` and `guard.py` may request `split_rngs("test", ...)` | static test in `tests/test_integrity.py` |
| Engine neutrality | `evaluation/engine.py` and `benchmarks/runner.py` contain no policy names | `tests/test_evaluation.py` |
| Historical results | SHA-256 of every result artifact is pinned | `tests/test_reproducibility.py` |
| Generated documents | `results/tables.md`, `docs/FINAL_RESEARCH_REPORT.md` and similar must equal their generator's output | `tests/test_integrity.py`; CI runs `execshift report --study && git diff --exit-code` |
| README numbers | headline values must match `authoritative.json`, written with ASCII hyphens | `tests/test_integrity.py` |
| Labels | new runs are POST-STUDY / EXPLORATORY; configs cannot claim "pre-registered" | `experiments/config.py` |

The code style is compact and functional: NumPy arrays shaped `(replicates, episodes)` or
`(episodes,)`, small pure functions, docstrings that give the definition and the unit.
Match the surrounding module.

## 4. Tests and checks

```bash
make check        # ruff + pytest: 261 tests, under a minute
make typecheck    # mypy (needs mypy installed)
make cli-smoke    # CLI end to end, into a temporary run registry
execshift inspect --lock          # lock status
execshift reproduce --verify --strict   # committed results vs the reference copy
```

Tests marked `slow` need the git-ignored PPO checkpoints or raw outputs, and skip when those
are absent. A fresh clone runs everything else.

| File | Covers |
|---|---|
| `test_core.py` | the frozen simulator, policies, statistics and contamination guard |
| `test_market.py` | scenarios reproduce the study's conditions; impact and latency invariants |
| `test_agents.py` | policy adapters are bit-identical to the frozen study policies; gated rollout equals the analytical gate |
| `test_uncertainty.py` | uncertainty estimators equal the frozen gate-feature columns they wrap |
| `test_metrics.py` | hand-computed values, agreement with the frozen study code, units |
| `test_statistics.py` | paired vs unpaired designs, practical verdicts, the frozen SESOI |
| `test_evaluation.py` | the policy-agnostic evaluation engine |
| `test_benchmarks.py`, `test_experiments.py`, `test_cli.py` | suites and recorded runs; config, hashing, manifests and provenance; the CLI |
| `test_reporting.py` | figures and reports are generated from artifacts and labelled; supplementary data agrees with the study |
| `test_integrity.py` | the study's safeguards: lock, test-split access, document staleness |
| `test_reproducibility.py` | frozen code, pinned historical artifacts, deterministic PPO training, regression pins |

## 5. Workflow for a change

1. Branch from `main`. CI runs on pull requests and on pushes to `main`.
2. Make the change in a platform subpackage, with a test that fails without it.
3. Run `make check typecheck` and `execshift inspect --lock`.
4. If the change affects a generated document, run `execshift report --study` and commit the
   regenerated file with an explanation.
5. Open a pull request using the checklist in `CONTRIBUTING.md` §4.

## 6. Adding a baseline policy

1. Write a class with a `name` attribute and `reset(ex)` and `act(ex) -> (B,)` sell fractions
   (`adexec/interfaces.py::ExecutionPolicy`). It may read only `ex.obs()`, `ex.q` and `ex.t`,
   never latent simulator state.
2. Register a `kind` in `agents/factory.py::make_policy` and add it to `POLICY_KINDS`.
3. Test it in `tests/test_agents.py`. If it wraps a frozen policy, check numerical equality.
4. Use it from a TOML config (`[[policies]] kind = "..."`) or a benchmark suite. It will be
   evaluated on the post-study stream (seed tag 66), never on the study's test split.

## 7. Adding a metric

1. Write a function in `adexec/evaluation/metrics.py` with a docstring that gives the
   definition, unit and interpretation.
2. Add a `Metric(name, unit, higher_is_better, fn, description)` entry to `METRICS`.
3. Add a hand-computed unit test in `tests/test_metrics.py`. If the frozen analysis has its
   own copy of the metric, test that the two agree.
4. Document it in [`METHODOLOGY.md`](METHODOLOGY.md) §7.

## 8. Running an experiment

```bash
execshift benchmark --list                    # standard suites
execshift benchmark --suite shift             # needs PPO checkpoints (make train)
execshift evaluate --config configs/examples/evaluate_liquidity.toml   # no PPO needed
execshift inspect <run-id>                    # manifest + artifact integrity
execshift evaluate --experiment <run-id>      # re-run and compare with the recorded numbers
```

Runs are written to `runs/<id>/` (git-ignored, or `$ADEXEC_RUNS_DIR`). Each has a manifest
with the git commit, config hash, seeds, environment and artifact hashes. Report every run
you make on a question, including the ones that disagree with you
(`CONTRIBUTING.md` §5).

## 9. Artifacts: what to commit

| Commit | Do not commit |
|---|---|
| source, tests, docs, configs | `models/`, `models_fast/` (checkpoints; weight hashes are in `results/rl_manifest.json`) |
| regenerated documents, with an explanation | `results/raw/`, `results/fast/` (regenerable, about 340 MB) |
| small post-study reductions (`results/post_study/`) and their figures | `runs/` (local experiment registry) |
| | any file over 1 MB, credentials, absolute paths |

## 10. Validating a change without corrupting results

- Never run `make reproduce-paper` or `execshift reproduce --full` casually. It deletes and
  regenerates `models/` and `results/raw/`, rewrites `results/authoritative.json` and appends
  780 lines to the test-access log. On the same platform the outputs are byte-identical, but
  the log growth is permanent.
- Use `make reproduce-paper-fast` to exercise every stage. It writes only `results/fast/` and
  `models_fast/` and never touches the test split.
- After any change, `pytest` checks the pinned hashes of every historical artifact. If one
  fails, a result file was modified: restore it with `git checkout -- <file>`.
- `execshift reproduce --verify --strict` confirms that `results/authoritative.json` still
  equals the reference copy.
