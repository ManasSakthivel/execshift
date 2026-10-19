# Architecture audit (Phase 0, pre-refactor)

*Written 2026-10-03, before any code was changed, against commit `5c08c6e`. This document
records the repository as it was found. The refactor that followed is described in
[`ARCHITECTURE.md`](ARCHITECTURE.md); this audit is kept unchanged as the "before" picture.*

## 0. The constraint that shapes everything

`research/prereg.lock` records SHA-256 hashes of the pre-registration and of **ten files**:

```
research/PREREGISTRATION.md
adexec/config.py   adexec/sim.py    adexec/policies.py  adexec/rl.py     adexec/features.py
adexec/gates.py    adexec/stats.py  adexec/analysis.py  adexec/tuning.py
```

`adexec/guard.py::check_lock()` refuses to generate test-split episodes if any byte of these
files changes, and CI and the integrity tests run the same check. These files hold the
simulator, the policies, the PPO set-up, the gate features, the gates, the statistics and
the hypothesis analysis. **They cannot be refactored, reformatted or re-linted** without
invalidating the confirmatory study. Editing them would require a recorded deviation and a
re-lock, which the brief forbids.

So the refactor cannot follow the usual approach of splitting large modules into packages.
It has to:

1. leave the ten frozen files byte-identical,
2. build typed interfaces, configuration, registry, benchmark, metrics, CLI and reporting
   layers **around** them, and
3. move only the non-frozen support code (runner, reports, figures, scripts).

## 1. Current architecture

```
adexec/                     (single flat package, ~2.9k lines)
  config.py      FROZEN  market parameters, 12 shift conditions, splits, mixture weights
  sim.py         FROZEN  market generator, causal features, vectorised execution engine
  policies.py    FROZEN  TWAP, Almgren-Chriss, ADAPT, Blend (callables ex -> v)
  rl.py          FROZEN  SB3 VecEnv, PPO hyper-parameters, member training, PPOEnsemble
  features.py    FROZEN  gate features: market summary, OOD, ensemble disagreement, shadow
  gates.py       FROZEN  RCG + threshold gates + controls, LOFO fitting, apply_gates
  stats.py       FROZEN  crossed bootstrap, Holm, Wilcoxon, AUROC/AUPRC, risk-coverage
  analysis.py    FROZEN  H1-H6 analysis, describe(), compare(), mixture tables
  tuning.py      FROZEN  AC / ADAPT tuning on the train split
  guard.py               split access, test-set guard, lock read/write, ROOT path
  experiment.py          evaluation runner (policy set hard-coded), pickle I/O, env-var dirs
  stylized.py            stylised-fact validation checks
  report.py / report_sim.py / report_tables.py   Markdown renderers (905 + 138 + 227 lines)
  plotstyle.py           matplotlib style
scripts/                 11 stage scripts (not importable; contain real logic)
tests/                   35 tests: test_core.py (14), test_integrity.py (21)
research/                pre-registration + lock
results/                 authoritative JSON, baselines, manifests, test-access log
                         (+ results/raw/*.pkl, 343 MB, git-ignored)
figures/                 10 PNGs
models/, models_fast/    PPO checkpoints (git-ignored; weight hashes in rl_manifest.json)
scratch/                 development scripts and logs (git-ignored)
```

### Dependency graph (imports)

```
config ─┬─> sim ─┬─> policies ─┬─> rl ──────────────┐
        │        │             └─> tuning <─ guard   │
        │        ├─> stylized                        │
        │        └─> features ──> gates ──> analysis <── stats
        └─> guard ──> experiment (also imports features, gates, policies, rl)
report.py ──> report_tables, guard, config
scripts/* ──> everything above (scripts are leaves; nothing imports them)
```

There are no import cycles. `guard` is the only module both the frozen core (`tuning`) and
the runner depend on. It also owns `ROOT`, the repository root derived from `__file__`.

## 2. Data flow

```
SeedSequence([split_tag, condition_index, episode])        (split tags 11/22/33/44/55)
        │
        ▼
generate_market(MarketParams, rngs) ──> Market (exogenous paths, B episodes)
        │                                   │
        │                                   ├──> market_features (causal EWMA features)
        ▼                                   ▼
Execution(m).step(v) ... ───────> IS (realised), IS_cv (control variate), trades
```

The exogenous market depends only on the episode seed. Every policy therefore faces the
same paths (common random numbers), which makes per-episode paired differences and
counterfactual regret exact.

## 3. Training flow

```
scripts/tune_baselines.py ─> tuning.tune_all(im) ─> train split (4,000 ID) ─> results/baselines_<im>.json
scripts/train_rl.py ─> rl.train_member(r, j, im) x 150 ─> models/<im>_r<rr>_m<j>.zip
                       ExecVecEnv: rltrain stream (tag 44), seed 5r+j, 20M steps, denoised reward
                     ─> results/rl_manifest.json (weights SHA-256)
```

## 4. Evaluation flow

```
scripts/run_validation.py ─> experiment.evaluate("val", im, 13 conds)
    per replicate (process pool): build the FIXED policy set
      {FB=TWAP, AC, ADAPT, RL1, RL, BLEND_0.25/0.5/0.75}, roll each out, compute gate features
    ─> results/raw/val_<im>.pkl
scripts/run_confirmatory.py (ADEXEC_CONFIRMATORY=1, lock checked)
    ─> experiment.evaluate("test", ...) x (3 impact models + 3 latencies) ─> results/raw/test_*.pkl
    ─> analysis.analyze(val, test): fit gates on val (LOFO), apply to test, H1-H6
    ─> results/authoritative.json
```

Gated policies are never rolled out. Their cost is computed **analytically** as
`np.where(fallback, IS_FB, IS_RL)`. This is valid because the gate decides once per episode
at arrival and episodes are independent. It is sound, but no test checked it.

## 5. Artifact flow

```
authoritative.json ─┬─> scripts/figure_data.py (needs results/raw) ─> figure_data.json
                    ├─> scripts/make_figures.py ─> figures/fig2..fig10
                    └─> scripts/build_report.py ─> docs/FINAL_RESEARCH_REPORT.md, results/tables.md,
                                                    FINAL_STANFORD_READINESS_REPORT.md,
                                                    docs/SIMULATOR_VALIDATION.md, docs/BASELINES.md
scripts/verify_reproduction.py reference.json new.json  (tolerance check)
tests/test_integrity.py: report/tables must equal what the renderers produce; README must
                         contain the headline numbers
```

## 6. Technical debt

| # | Finding | Where | Severity |
|---|---|---|---|
| D1 | **Policy-specific logic in the evaluation runner.** `_eval_replicate` hard-codes which policies exist and how each is built, so a new policy needs runner edits. | `experiment.py` | high |
| D2 | **No explicit policy interface.** Policies are callables `policy(ex)` with an optional `reset`, found by `hasattr`, and the gate is not a policy at all. | `policies.py`, `rl.py`, `sim.rollout` | high |
| D3 | **Import-time global state.** `MODELS`, `RAW` and `BASELINES_DIR` are read from environment variables when `experiment` is imported. `reproduce_fast.py` must set them before its imports (`# noqa: E402`), so import order silently decides where artifacts go. | `experiment.py`, `scripts/reproduce_fast.py` | high |
| D4 | **cwd-dependent paths.** `validate_simulator.py`, `tune_baselines.py` and `train_rl.py` write to the relative path `results/` and `figures/`. Run from any other directory, they scatter or misplace artifacts. | `scripts/` | medium |
| D5 | **`ROOT` from `__file__`.** It is correct for an editable install. A regular `pip install` puts the package in site-packages, and every path (lock, results, access log) then points to the wrong place. | `guard.py` | medium |
| D6 | **Logic lives in non-importable scripts.** Figure data, figures, simulator-validation figure and the fast pipeline can only be run as `python scripts/x.py`, so a CLI or test cannot call them. | `scripts/` | medium |
| D7 | **Duplicated metric code.** CVaR (`analysis.describe`), weighted quantiles (`figure_data._wq`), Spearman/AUROC (`analysis._h3_metrics`, `_h3_core`, `figure_data`) and three bootstrap loops each with their own seeds. The frozen copies cannot be removed. | `analysis.py`, `figure_data.py` | medium |
| D8 | **Parameters scattered across code.** Configuration is in Python constants spread over 9 modules and scripts (`LATENCIES` in `run_confirmatory.py`; `REPS`, `STEPS`, `N_EP` in `reproduce_fast.py`). There is no machine-readable experiment config and no config hash. | everywhere | medium |
| D9 | **No per-run manifest.** `authoritative.json` records the prereg hash and lock time, but not the git commit, environment or config hash. The environment appears only as prose in `status.json`. | `results/` | medium |
| D10 | **The verifier checks a subset.** `verify_reproduction.py` compares 1,063 of 14,378 numeric leaves (estimates, CI bounds, decisions). It does **not** check the H3 point metrics (Spearman ρ, AUROC). `zip(..., strict=False)` silently ignores lists of different length. The committed same-platform reproduction was byte-identical, so it is unaffected. A cross-platform run would not verify H3. | `scripts/verify_reproduction.py` | medium |
| D11 | **Pickle for raw outputs.** 343 MB of pickles, unversioned schema, unsafe to load from untrusted sources. They are local only and never committed. | `experiment.py` | low |
| D12 | **CI does too much on every push and too little across platforms.** Every push runs the reduced pipeline. There is one Python version, no type check, and no nightly or release separation. | `.github/workflows/ci.yml` | low |
| D13 | **Test gaps.** No tests for: the analytical gate shortcut (D2), figure or report generation beyond staleness, the verifier, CLI, configuration, or engine invariants across many seeds (latency, impact monotonicity). | `tests/` | medium |
| D14 | **Dev tooling is not separated from the runtime lock.** `pytest` and `ruff` are pinned inside `requirements-lock.txt`, the record of the study environment. | `requirements-lock.txt` | low |

### Undocumented assumptions found

- **A1.** Gated cost via `np.where` assumes once-per-episode decisions and per-episode
  independence (D2). True by construction, untested.
- **A2.** `experiment._eval_replicate` regenerates the *train* market for every replicate
  to fit the OOD model. It is deterministic, so this is correct, but redundant work.
- **A3.** `ACTION_LOG_RANGE = ln 4` clips both ADAPT and RL. The fallback headroom
  discussion in the report depends on it. It is documented in the report's limitations but
  not near the code that uses it.
- **A4.** Single-threaded torch (`torch.set_num_threads(1)`) is what makes training
  bit-for-bit deterministic. This was stated in `docs/REPRODUCIBILITY.md` but nothing
  enforced or tested it.

## 7. Scientific issues discovered (documented, not "fixed")

These do not change any historical number. Following the brief, they are recorded here and in
[`LIMITATIONS.md`](LIMITATIONS.md) rather than silently corrected.

- **S1. Pre-registration text and locked code disagree on the ADAPT grid.**
  `PREREGISTRATION.md` §6.1 says ADAPT weights come from "three rounds of coordinate
  descent on a 17-point grid in [-2, 2]". The locked `adexec/tuning.py` (hash recorded in
  the same lock) uses a **25-point grid on [-3, 3]**, and `results/baselines_*.json` show
  that grid was used. `docs/BASELINES.md` says the grid was widened twice "before any RL
  policy was evaluated", to keep the optima interior. The code that ran is the locked code,
  so the confirmatory result is internally consistent. But the pre-registration prose is
  stale on this point, and §11 "Deviations" records nothing. Effect on conclusions: none
  that this audit can identify. A wider grid can only help the baseline that RL is compared
  against, which makes H1 harder to pass, not easier. It is still a documentation defect in
  a frozen document.
- **S2. Git history cannot independently corroborate the pre-lock timeline.** The whole
  development history was committed in one commit (`234759f`), 7 seconds before the lock
  commit (`f84f3b3`). The order in the pre-registration timeline (shift suite frozen before
  RL training, development on validation only) rests on the authors' account and on the
  test-access log. The log is append-only by convention but not tamper-evident: it is
  written by the same code it audits. The *test* split's integrity is better supported. It
  is gated by a hash check, every access is logged after the lock time, and the static test
  forbids development code from requesting it.
- **S3. Verifier coverage (D10).** "1,063 checks, 0 outside tolerance" is accurate but does
  not cover the H3 metrics. The same-platform claim of byte-identity does.

## 8. Proposed architecture

Principle: **the frozen core stays as it is at the top of `adexec/`. Everything new goes in
subpackages that depend on it, and nothing in the core depends on them.**

```
adexec/
  {config,sim,policies,rl,features,gates,stats,analysis,tuning}.py   FROZEN core (unchanged)
  guard.py                     split access + test guard (kept in place: frozen docs cite it)
  paths.py                     single source of truth for repository/artifact locations
  interfaces.py                typing.Protocols for the platform's extension points
  market/                      ImpactModel / LatencyModel specs, Scenario -> MarketParams,
                               MarketSimulator adapter, stylised-fact validation
  agents/                      ExecutionPolicy adapters for TWAP, AC, ADAPT, PPO,
                               PPO-ensemble and the uncertainty-gated policy; a policy factory
                               ("policies" is taken by the frozen module)
  uncertainty/                 UncertaintyEstimator adapters around features.py
  evaluation/                  policy-agnostic rollout engine, metrics engine,
                               statistics layer (paired/unpaired, CIs, practical verdicts)
  benchmarks/                  declarative suites (baseline, ID, shift, impact, latency,
                               liquidity, risk) + runner
  experiments/                 TOML config + hashing, run manifests, ArtifactStore,
                               provenance lookup for the study's published numbers
  pipeline/                    the study pipeline stages, moved out of scripts/ (scripts
                               stay as thin entry points because frozen docs cite them)
  reporting/                   report renderers, figure style, study figures, new
                               supplementary figures
  cli/                         `execshift` command (argparse, calls the library)
```

Why not a `src/` layout: the lock records paths as `adexec/<file>.py` relative to the
repository root, and `guard` resolves them against `ROOT`. Moving the package would break
every lock check, or force a re-lock. A re-lock is a protocol deviation.

## 9. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A frozen file is modified (even by an auto-formatter) | medium | critical: invalidates the study | Exclude frozen files from `ruff format`. Lock check runs in tests and CI. A new test compares the frozen file list with the lock. |
| Moving the runner, reports or figures changes historical outputs | medium | high | Byte-compare regenerated `figure_data.json`, figures, reports and tables against the committed files. Rerun the full pipeline in a clean clone. |
| New benchmark numbers mistaken for pre-registered results | medium | high | Separate seed stream (tag 66) that the study never used. Every manifest carries `label: POST-STUDY / EXPLORATORY`. Separate output directory (`runs/`). Banner in every benchmark report. |
| Test split touched by new code | low | critical | New code never calls `split_rngs("test")`. The static integrity test is extended to the new packages. |
| Abstraction for its own sake | medium | medium | Each interface must have at least two real implementations or one real consumer (the engine, CLI or benchmarks). Anything else is cut. |
| Cross-platform nondeterminism read as reproduction failure | medium | medium | The verifier is extended, not loosened. Pre-declared tolerances are unchanged. Platform assumptions are documented. |

## 10. Addendum: second-pass forensics of the uncommitted refactor (2026-10-03)

*Added in a second session, which started from an uncommitted working tree on top of
`17e688d`. Sections 0-9 above are unchanged.*

**State found.** `git status` showed a refactor partway through, following §8:

| Change | State | Verified |
|---|---|---|
| `paths.py`, `interfaces.py` | new, complete | `guard.py` now takes `ROOT` from `paths`; lock entries resolve to the imported package (`guard.frozen_path`) |
| `market/` (simulator adapter, impact/latency specs, `Scenario`, stream 66; `stylized.py` → `market/validation.py`) | complete | `Scenario.params()` equals the pre-refactor `condition_params` for 13 conditions × 3 impact models × 4 latency settings |
| `agents/` (TWAP, AC, ADAPT, Blend, PPO, `GatedPolicy`, factory) | complete | sell quantities bit-identical to the frozen classes |
| `uncertainty/` (ensemble, OOD, shadow, RCG adapters) | complete | scores equal the frozen gate-feature columns |
| `evaluation/` (engine, metrics, statistics) | complete | engine equals `sim.rollout`; metrics agree with the frozen versions |
| `pipeline/` (all stages moved out of `scripts/`; `experiment.py` → `pipeline/evaluation.py`) | complete; scripts reduced to thin entry points | regenerated documents, tables and figures byte-identical |
| `reporting/` (`report*.py`, `plotstyle.py` moved) | complete | report/table staleness tests pass |
| `benchmarks/`, `cli/`, `experiments/` | **empty directories** | referenced by `interfaces.py` and by the edited `tests/test_integrity.py` (`adexec/benchmarks/runner.py`), but not yet written |
| tests named in docstrings (`test_agents.py`, `test_market.py`, ...) | **missing** | the docstrings cited tests that did not exist |
| `pipeline/verify.py` | post-study hardening: list-length mismatches fail; supplementary coverage report; `--strict` | stricter only; the pre-declared tolerance is unchanged |

At session start the test suite had **1 failure**: `test_static_scan_covers_the_codebase`
expected `adexec/benchmarks/runner.py`. The lock check passed, and `results/` and `research/`
were byte-identical to `HEAD`.

**What the second session did.** It kept all of the above and built on it: the missing
`experiments/`, `benchmarks/` and `cli/` layers, the missing tests, the supplementary
figures, CI and documentation. Nothing was reset, stashed or deleted. Two small additive
edits were made to existing refactor files: `agents/` gained an `ensemble_size` option and a
post-study trainer, and `benchmarks/runner.py` reads the gate diagnostic without
`isinstance`. One bug was found and fixed by type checking (`execshift report --study` called
`main` on a function).

**Additional finding (security / provenance).** The local repository still has the upstream
clone's `master` branch and the `upstream` remote. They contain the unlicensed upstream code,
including an Alpaca trading adapter that takes `api_key` / `secret_key` parameters (no actual
keys). None of it is reachable from `main`. Pushing with `git push --all`, or pushing
`master`, would publish unlicensed third-party code. Push `main` only. Deleting the branch
and the remote is left to the owner.

Audit items resolved by the refactor: D1-D9, D10 (verifier hardened), D12 (CI split into
PR / nightly / release), D13 (tests added). Still open: D11 (pickled raw outputs, local
only) and D14 (dev tools pinned in the runtime lock; changing the lock file would rewrite
the study's environment record).
