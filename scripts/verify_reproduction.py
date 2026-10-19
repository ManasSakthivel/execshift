"""Compare a regenerated authoritative.json with the committed reference.

Thin entry point; the implementation is ``adexec.pipeline.verify``."""

from adexec.pipeline.verify import main

if __name__ == "__main__":
    raise SystemExit(main())
