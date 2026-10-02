# Provenance and licensing

## Where this repository came from

This working copy started as a clone of an upstream repository:

| | |
|---|---|
| Upstream | [`Bilal-Aamir-Yousuf/rl-market-maker`](https://github.com/Bilal-Aamir-Yousuf/rl-market-maker) |
| Author | Bilal Aamir Yousuf (all 13 upstream commits, 6 June 2026) |
| Upstream HEAD | `c392a4f` ("docs: rewrite README prose to read more naturally") |
| Subject | An RL **market maker** (PPO/SAC/TD3/DDPG) on a Hawkes-driven limit order book, benchmarked against Avellaneda-Stoikov, with an Alpaca paper-trading layer |
| License | **None.** The repository has no `LICENSE` file, and GitHub reports `licenseInfo: null`. |

The upstream remote is kept as `upstream` in the local clone. Its full history is preserved
untouched on the local `master` branch.

## What we may and may not redistribute

A public repository with no license is, by default, **all rights reserved**. GitHub's
Terms of Service let other users view and fork it on GitHub. They do not grant a licence to
copy, modify, or redistribute the code anywhere else.

So this project:

1. **Does not contain or redistribute any upstream code, text, figures, or data.** This
   branch (`main`) is an orphan branch, started from an empty tree. No file in it was copied
   or adapted from upstream.
2. **Does not reuse upstream algorithms.** The upstream components were a limit order book
   matching engine, a Hawkes order-flow simulator, a market-making Gymnasium environment, an
   Avellaneda-Stoikov baseline, and paired significance tests for market making. They solve
   a different problem (two-sided quoting), and none of them is used here.
3. **Does not represent upstream work as ours, or ours as upstream's.** The project has a
   new name (`adexec`) because it is a different research project on a different problem,
   not a rebrand of the upstream implementation.

The task description for this project characterised the upstream as "adaptive execution
using RL and switching/gating between execution policies". When we inspected the upstream
code, that turned out to be inaccurate. The upstream contains no execution policies (TWAP,
Almgren-Chriss), no gating, no rule gate, no ensemble control, and no random-switching
control. Every such component here (including the controls labelled A3, A9, and A10) is new.

## Upstream contribution and our contribution

### Upstream contribution (cited, not included)

- A from-scratch price-time-priority LOB matching engine.
- A Hawkes order-flow market-making simulator.
- An RL market maker compared with Avellaneda-Stoikov (2008).
- An Alpaca paper-trading adapter.

None of these appears in this repository. The upstream project informed our choice of
subject area (RL in market microstructure) and the general habit of paired, multi-seed
evaluation. Paired evaluation is standard practice and not specific to the upstream.

### Our contribution (everything in this repository)

| Component | Files | Status |
|---|---|---|
| Synthetic execution simulator: stochastic volatility, Student-t returns, latent alpha with a noisy signal, spread/depth/volume state processes, three impact models (square-root concave, linear AC, transient propagator), latency via a Brownian bridge | `adexec/sim.py`, `adexec/config.py` | new |
| Stylised-fact validation suite (14 checks per impact model) | `adexec/stylized.py`, `scripts/validate_simulator.py` | new |
| Classical baselines: TWAP, AC (incl. mirrored back-loading), tuned opportunistic ADAPT, blend | `adexec/policies.py`, `adexec/tuning.py` | new |
| Batched SB3 `VecEnv`, PPO deep ensembles, control-variate training reward | `adexec/rl.py` | new (uses SB3, MIT) |
| Gate features (pre-trade summary, Mahalanobis OOD, ensemble disagreement, shadow-regret replay) | `adexec/features.py` | new |
| Regret-calibrated gate (RCG), LOFO fitting, A3/A9/A10 and ablation controls | `adexec/gates.py` | new |
| Crossed (two-way) bootstrap, Holm, risk-coverage, AUROC/AUPRC | `adexec/stats.py`, `adexec/analysis.py` | new |
| Pre-registration with a hash-locked test guard and access log | `research/`, `adexec/guard.py` | new |
| Experiments, figures, report generation, reproduction pipeline, integrity tests, CI | `scripts/`, `tests/`, `Makefile`, `.github/` | new |

## Third-party software

All third-party code is used as an unmodified dependency and installed from PyPI. None of
it is vendored into this repository.

| Package | License |
|---|---|
| NumPy, SciPy, pandas, scikit-learn | BSD-3-Clause |
| PyTorch | BSD-style |
| Stable-Baselines3 | MIT |
| Gymnasium | MIT |
| Matplotlib | Matplotlib License (PSF-based, BSD-compatible) |

## License of this repository

The new code and documentation in this repository are released under the MIT License (see
`LICENSE`). This license does **not** apply to the upstream repository, which remains the
property of its author under its own (absent) licence terms.

## Data provenance

There is no external market data. Every market path is synthetic, generated from
`numpy.random.SeedSequence([split_tag, condition_index, episode_index])`, so any episode can
be regenerated exactly from these three integers.

| Split | Tag | Use |
|---|---|---|
| train | 11 | baseline tuning; OOD and threshold reference distributions |
| val | 22 | PPO hyperparameter selection (ID only); fitting every gate and control |
| test | 33 | confirmatory evaluation, once, after the pre-registration lock |
| rltrain | 44 | unbounded stream of ID episodes for PPO training |
| simval | 55 | simulator validation |

The simulator's parameters come from published empirical ranges, listed in
`docs/SIMULATOR_VALIDATION.md`. They are not fitted to any asset, venue, or period.
