# Reproduction pipeline.  All data are synthetic and generated from seeds.
#
#   make reproduce-paper       full paper (76.5 min wall on 14 cores; ~10 CPU-hours)
#   make reproduce-paper-fast  reduced sanity pipeline (~1-2 min); never touches the test split
#   make check                 lint + 258 tests (~15 s);  make typecheck  mypy (platform layers)
#
# The same stages are available as `execshift reproduce --stage <name>` (see docs/EXPERIMENTS.md).
#
# The confirmatory stage requires research/prereg.lock to match the frozen code; it appends
# to results/test_access_log.jsonl.  verify compares the regenerated results with the
# committed reference copy (results/authoritative.reference.json) within the tolerance
# documented in scripts/verify_reproduction.py.

PY      ?= .venv/bin/python
WORKERS ?= $(shell python3 -c "import os; print(max(1, (os.cpu_count() or 2) - 1))")

.PHONY: install validate-sim tune train val confirm figures report verify test lint \
        typecheck check cli-smoke supplementary reproduce-paper reproduce-paper-fast \
        clean-generated

.venv/bin/python:
	@if command -v uv >/dev/null 2>&1; then \
	  uv venv --python 3.12 .venv && \
	  uv pip install --python .venv/bin/python -r requirements-lock.txt && \
	  uv pip install --python .venv/bin/python -e . --no-deps; \
	else \
	  python3.12 -m venv .venv && \
	  .venv/bin/python -m pip install -r requirements-lock.txt && \
	  .venv/bin/python -m pip install -e . --no-deps; \
	fi

install: .venv/bin/python

validate-sim: install          ## 1. simulator validation (stylised facts)
	$(PY) scripts/validate_simulator.py

tune: install                  ## 2. baseline tuning (train split)
	$(PY) scripts/tune_baselines.py

train: install                 ## 3. 150 PPO members (3 impact models x 10 replicates x 5)
	$(PY) scripts/train_rl.py --impact sqrt linear transient --workers $(WORKERS)

val: install                   ## 4. validation-split evaluation + gate features
	$(PY) scripts/run_validation.py --workers $(WORKERS)

confirm: install               ## 5. confirmatory test-split evaluation + statistics
	$(PY) scripts/run_confirmatory.py --workers $(WORKERS)

figures: install               ## 6. figures
	$(PY) scripts/figure_data.py
	$(PY) scripts/make_figures.py

report: install                ## 7. tables, simulator-validation doc and research report
	$(PY) scripts/build_report.py

verify: install
	$(PY) scripts/verify_reproduction.py results/authoritative.reference.json results/authoritative.json

test: install
	$(PY) -m pytest

lint: install
	$(PY) -m ruff check .

typecheck: install             ## mypy on the post-study platform layers (needs `pip install mypy`)
	$(PY) -m mypy

check: lint test               ## what PR CI runs (plus typecheck)

cli-smoke: install             ## POST-STUDY: CLI end to end into a throw-away registry
	ADEXEC_RUNS_DIR=$$(mktemp -d) sh -c '$(PY) -m adexec benchmark --suite smoke --quiet && \
	  $(PY) -m adexec inspect && $(PY) -m adexec inspect --claim H4 >/dev/null'

supplementary: install         ## POST-STUDY descriptive figures S1-S7 (S4-S7 need results/raw)
	$(PY) -m adexec.reporting.supplementary

clean-generated:
	rm -rf models results/raw

reproduce-paper: clean-generated validate-sim tune train val confirm figures report verify test
	@echo "reproduce-paper: done"

reproduce-paper-fast: install
	$(PY) scripts/reproduce_fast.py
