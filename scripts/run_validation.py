"""Stage 4: evaluate every policy and compute gate features on the VALIDATION split.

Thin entry point; the implementation is ``adexec.pipeline.validation``."""

from adexec.pipeline.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
