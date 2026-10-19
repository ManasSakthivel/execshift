"""Stage 5: the confirmatory run (TEST split, requires the pre-registration lock).

Thin entry point; the implementation is ``adexec.pipeline.confirmatory``."""

from adexec.pipeline.confirmatory import main

if __name__ == "__main__":
    raise SystemExit(main())
