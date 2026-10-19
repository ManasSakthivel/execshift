"""Stage 7: render every generated document from results/*.json.

Thin entry point; the implementation is ``adexec.pipeline.documents``."""

from adexec.pipeline.documents import main

if __name__ == "__main__":
    raise SystemExit(main())
