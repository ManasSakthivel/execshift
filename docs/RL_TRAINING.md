# RL training: design, hyperparameter search and budget

## Setup

| | |
|---|---|
| Algorithm | PPO (Stable-Baselines3 2.x), `MlpPolicy`, separate 64-64 tanh actor and critic |
| Environment | `adexec.rl.ExecVecEnv`: a synchronous batch of 128 in-distribution episodes, all of the same length (60 steps) |
| Training data | an unbounded stream of ID episodes from split tag 44 (`rltrain`), disjoint from train, val, test and simval. Each member has its own sub-stream (`stream_id = impact*1000 + seed`). |
| Observation (9-d) | inventory fraction, time remaining, signal, log realised vol, log spread, log depth, log volume, last return / nominal σ, log order size. The trader's nominal references are used throughout. |
| Action | a ∈ [−1, 1] → sell exp(a·ln 4) × the remaining-TWAP pace (a = 0 reproduces TWAP); the final step liquidates |
| Reward | per-step mark-to-market P&L in bp. Episode rewards sum to −IS. |
| Training-only variance reduction | the exogenous return term is replaced by its conditional mean given the simulator state (`denoise=True`). Expected returns are unchanged (tower property). Evaluation always uses realised IS. |
| Deployed RL | mean action of a 5-member deep ensemble |
| Replicates | 10 per impact model (member seed = 5·replicate + member) |
| Checkpointing | the final checkpoint is used; there is no checkpoint selection |
| Determinism | `torch.set_num_threads(1)` with a fixed seed. Bit-for-bit reproducible on the same platform. |

## Hyperparameter search

**Data used:** the 2,000 in-distribution validation episodes only. No shifted validation
conditions and no test data were used.

**Metric:** mean paired IS difference against TWAP (lower is better). Each configuration is
a single member (seed 777 or 12345). We report every configuration tried, including those
that did worse.

| Config | Change from defaults (`n_steps`=120, 128 envs, batch 1024, 5 epochs, lr 3e-4, GAE 0.95, log-std −1) | Steps | RL − TWAP (bp) | SE |
|---|---|---|---|---|
| dev-0 | no reward denoising | 3M | −0.773 | 0.089 |
| dev-1 | reward denoising | 3M | −1.059 | 0.252 |
| A | — (denoising on from here on) | 20M | −2.262 | 0.160 |
| B | lr 1e-4, 10 epochs | 20M | −1.824 | 0.223 |
| C | no denoising | 20M | −1.292 | 0.180 |
| D | batch 4096, GAE 0.98 | 20M | −2.449 | 0.167 |
| E | 128-128 networks | 20M | −1.557 | 0.210 |
| F | 10 epochs, batch 2048 | 20M | −2.064 | 0.142 |
| G | lr 1e-3 | 20M | −1.047 | 0.124 |
| H | log-std init −2 | 20M | −0.450 | 0.046 |
| D2 | as D (new seed) | 50M | −2.186 | 0.163 |
| D3 | D + GAE 1.0 | 50M | −2.160 | 0.182 |
| D4 | D + batch 8192 | 50M | −2.433 | 0.184 |
| D5 | D + 10 epochs | 50M | −2.184 | 0.147 |
| D6 | D + 256 envs | 50M | −2.346 | 0.168 |
| **D7** | **D + lr 1e-4** | 50M | **−2.483** | 0.186 |

**Selected:** the D7 configuration (batch 4096, GAE 0.98, lr 1e-4), trained for **20M
steps** per member.
- D7's learning curve sits on its plateau from about 6M steps onward (−2.41 at 20M, −2.48
  at 50M), and the 50M gain is within one SE.
- The top configurations (D, D4, D7) are statistically indistinguishable. We picked the
  lower learning rate for stability, not for its point estimate.
- The search budget was 16 configurations, about 500M environment steps in total.

On the same validation episodes, the tuned ADAPT heuristic scores −2.44 bp against TWAP. A
single D7 member therefore ties a tuned feature-based heuristic in-distribution; it does
not dominate it. We report this as is.

## Final training budget

| | |
|---|---|
| Members | 150 = 3 impact models × 10 replicates × 5 members |
| Steps per member | 20M (about 220 s on one Apple M4 Pro core) |
| Total | 3.0B environment steps; about 9 CPU-hours |

Checkpoint SHA-256 hashes are in `results/rl_manifest.json`. The checkpoints themselves are
not committed: `make train` regenerates them deterministically on the same platform.
