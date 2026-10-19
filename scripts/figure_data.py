"""Stage 6a: descriptive per-condition summaries -> results/figure_data.json.

Thin entry point; the implementation is ``adexec.pipeline.figure_data``."""

from adexec.pipeline.figure_data import main

if __name__ == "__main__":
    raise SystemExit(main())
