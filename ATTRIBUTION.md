# Attribution

This file says what this repository owes to others and what it does not. Detailed licensing
reasoning for the upstream clone is in [`PROVENANCE.md`](PROVENANCE.md). The full literature
review, with the adversarial novelty audit, is in
[`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) and
[`NOVELTY_STATEMENT.md`](NOVELTY_STATEMENT.md).

## 1. Upstream repository: cited, not included

| | |
|---|---|
| Repository | [`Bilal-Aamir-Yousuf/rl-market-maker`](https://github.com/Bilal-Aamir-Yousuf/rl-market-maker), by Bilal Aamir Yousuf |
| Subject | RL market making on a Hawkes-driven limit order book vs Avellaneda-Stoikov; Alpaca paper-trading adapter |
| License | **none** (all rights reserved by default) |
| Code from it in this repository | **none**. `main` is an orphan branch started from an empty tree. |
| Influence | choice of subject area (RL in market microstructure); the general habit of paired, multi-seed evaluation, which is standard practice |

The local clone still contains the upstream history on branch `master` and the remote
`upstream`. They are not part of `main`. **Publish `main` only**
(`git push origin main`, never `--all`), so the unlicensed upstream code is not
redistributed.

## 2. Conceptual sources implemented here

Each of these is implemented from the published description. No reference code was used.

| Concept | Source | Where |
|---|---|---|
| Optimal execution trajectory | Almgren & Chriss (2001), *J. Risk* | `policies.AlmgrenChriss` |
| Square-root (concave) impact | Tóth et al. (2011), *PRX*; Donier, Bonart, Mastromatteo & Bouchaud (2015) | `sim.Execution` (`sqrt`) |
| Transient impact / propagator | Obizhaeva & Wang (2013), *J. Financial Markets*; Bouchaud et al. (2018) | `sim.Execution` (`transient`) |
| Stylised facts used for validation | Cont (2001), *Quantitative Finance* | `market/validation.py` |
| PPO | Schulman et al. (2017) | via Stable-Baselines3 |
| Deep ensembles for epistemic uncertainty | Lakshminarayanan et al. (2017), *NeurIPS* | `rl.PPOEnsemble`, `features.policy_uncertainty` |
| Mahalanobis OOD score | Lee et al. (2018), *NeurIPS* | `features.OODModel` |
| Selective prediction / risk-coverage | Geifman & El-Yaniv (2017); El-Yaniv & Wiener (2010) | `stats.risk_coverage`, `stats.aurc` |
| Control variates from approximating martingales | Henderson & Glynn (2002) | `sim.Execution.shortfall_cv_bps` |
| Crossed (pigeonhole) bootstrap | Owen (2007), *Ann. Appl. Stat.* | `stats.crossed_bootstrap` |
| Holm step-down correction | Holm (1979) | `stats.holm` |
| Seed variance in deep RL evaluation | Henderson et al. (2018); Agarwal et al. (2021) | 10 replicates, no best-run selection |

The gating *method* (deciding when to defer from a learned policy to a safe fallback) is
established in prior work: AAMAS-2025 evaluation-time switching, FineFT (2025), and Irshad &
Biswas (2026), among others. This repository claims only the empirical answer and the
benchmark, not the method (`NOVELTY_STATEMENT.md`).

## 3. Software dependencies

Installed from PyPI as unmodified dependencies. Nothing is vendored. Exact versions are in
`requirements-lock.txt`.

| Package | Use | License |
|---|---|---|
| NumPy, SciPy, pandas | numerics, statistics | BSD-3-Clause |
| scikit-learn | gate regressors, scaling | BSD-3-Clause |
| PyTorch | PPO networks (CPU) | BSD-style |
| Stable-Baselines3 | PPO implementation | MIT |
| Gymnasium | environment spaces | MIT |
| Matplotlib | figures | Matplotlib License (PSF-based) |
| pytest, ruff, mypy, pre-commit | development only | MIT |

## 4. Data

**No external data.** Every market path is synthetic, generated from seeds by
`adexec/sim.py`. Parameter values come from ranges reported in the empirical microstructure
literature (sources per parameter in `docs/SIMULATOR_VALIDATION.md`). They are not fitted to
any proprietary or licensed dataset.

## 5. Development assistance

The code and documents in this repository were developed with an AI coding assistant
(Anthropic's Claude, via Claude Code), as recorded in the `Co-Authored-By` trailer of every
commit. The research question, the pre-registration decisions and responsibility for every
claim belong to the author.

## 6. License of this repository

New code and documents: MIT ([`LICENSE`](LICENSE)). Citation metadata:
[`CITATION.cff`](CITATION.cff).
