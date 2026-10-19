"""Stage 1: simulator validation (stylised facts) -> results/simulator_validation.json + figure.

Thin entry point; the implementation is ``adexec.pipeline.simulator_validation``."""

from adexec.pipeline.simulator_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
