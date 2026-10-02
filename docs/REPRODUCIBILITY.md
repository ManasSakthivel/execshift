# Reproducibility

## Clean-clone reproduction (performed 2026-10-02)

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

## Tolerances and nondeterminism

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
