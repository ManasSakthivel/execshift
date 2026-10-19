"""make reproduce-paper-fast: reduced end-to-end sanity pipeline (never touches TEST).

Thin entry point; the implementation is ``adexec.pipeline.fast``."""

from adexec.pipeline.fast import main

if __name__ == "__main__":
    raise SystemExit(main())
