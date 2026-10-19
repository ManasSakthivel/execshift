"""Stage 2: tune AC and ADAPT on the TRAIN split -> results/baselines_<impact>.json.

Thin entry point; the implementation is ``adexec.pipeline.baselines``."""

from adexec.pipeline.baselines import main

if __name__ == "__main__":
    raise SystemExit(main())
