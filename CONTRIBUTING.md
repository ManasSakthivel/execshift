# Contributing

Thank you for considering a contribution. This repository is unusual in one respect:
part of it is a **frozen, pre-registered study**. Read section 1 before changing anything.

## 1. What you must not change

| Files | Why |
|---|---|
| `adexec/{config,sim,policies,rl,features,gates,stats,analysis,tuning}.py`, `research/PREREGISTRATION.md` | Hash-locked by `research/prereg.lock`. Any byte change (including auto-formatting) makes `check_lock()` refuse the confirmatory run. `ruff format` excludes them. Do not run other formatters on them. |
| `research/prereg.lock` | Re-locking is a protocol deviation and must be recorded as one. |
| `results/authoritative*.json`, `results/figure_data.json`, `results/tables.md`, `results/baselines_*.json`, `results/simulator_validation.json` | Historical results, pinned by SHA-256 in `tests/test_reproducibility.py`. |
| `results/test_access_log.jsonl` | Append-only audit trail. |

A change to the simulator's dynamics, the 12 shift definitions, the gates or the
pre-registered analysis is a **new study**: it needs a new pre-registration and a new
locked test stream, not a pull request against this one.

## 2. What you can extend freely

Policies, uncertainty estimators, metrics, scenarios, benchmark suites, simulators, CLI
commands, reports. See `docs/ARCHITECTURE.md` §10 for where each goes. New results must be
labelled **POST-STUDY / EXPLORATORY**: the config loader and the benchmark runner do this
automatically, and refuse "pre-registered" labels.

## 3. Development set-up

```bash
make install                    # .venv from requirements-lock.txt (Python 3.12) + editable install
.venv/bin/pip install mypy pre-commit && .venv/bin/pre-commit install   # optional
make check                      # ruff + pytest (under a minute)
make typecheck                  # mypy on the platform layers
make cli-smoke                  # CLI end to end into a temporary registry
```

## 4. Pull-request checklist

- [ ] `make check typecheck` passes; `execshift inspect --lock` reports OK.
- [ ] New behaviour has tests that would fail without it: no test-count padding, no tests
      that only assert "does not crash".
- [ ] New metrics: implementation + docstring (definition, unit, interpretation) + an
      entry in `METRICS` + a hand-computed unit test.
- [ ] New comparisons report an effect size and a practical verdict, never a p-value alone.
      Any threshold other than the pre-registered 0.25 bp carries a `threshold_source`.
- [ ] New scenarios outside the 12 pre-registered shifts are named `... (NEW)`.
- [ ] No code outside `adexec/pipeline/confirmatory.py` requests the test split (the
      static test enforces this).
- [ ] No absolute paths, user names, credentials or files over 1 MB.
- [ ] If generated documents change, regenerate them (`execshift report --study`) and
      explain why in the PR.

## 5. Reporting results

Report every run you make on a question, including those that disagree with the study or
with your hypothesis. Include the run id and config hash. `docs/BENCHMARKS.md` §3.4 shows
how to handle a disagreement.
