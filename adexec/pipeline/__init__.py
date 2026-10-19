"""The pre-registered study pipeline, as importable stages.

| Stage | Module                    | Script entry point (cited by the frozen docs) |
|-------|---------------------------|-----------------------------------------------|
| 1     | ``simulator_validation``  | ``scripts/validate_simulator.py``             |
| 2     | ``baselines``             | ``scripts/tune_baselines.py``                 |
| 3     | ``training``              | ``scripts/train_rl.py``                       |
| 4     | ``validation``            | ``scripts/run_validation.py``                 |
| 5     | ``confirmatory``          | ``scripts/run_confirmatory.py``               |
| 6a    | ``figure_data``           | ``scripts/figure_data.py``                    |
| 6b    | ``reporting.study_figures`` | ``scripts/make_figures.py``                 |
| 7     | ``documents``             | ``scripts/build_report.py``                   |
| 8     | ``verify``                | ``scripts/verify_reproduction.py``            |
| fast  | ``fast``                  | ``scripts/reproduce_fast.py``                 |

Every stage exposes ``main(argv=None) -> int``. The Makefile, the scripts and the CLI all call
these functions. The modules were moved out of ``scripts/`` without changing their
outputs, which the committed artifacts verify byte for byte.
"""
