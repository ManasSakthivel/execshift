# Reproduction pipeline.  All data are synthetic and generated from seeds.
#
#   make reproduce-paper       full paper (~2 h on a 14-core machine; ~9 CPU-hours of PPO)
#   make reproduce-paper-fast  reduced sanity pipeline (~5-10 min); never touches the test split
#
# The confirmatory stage requires research/prereg.lock to match the frozen code; it appends
# to results/test_access_log.jsonl.  verify compares the regenerated results with the
# committed reference copy (results/authoritative.reference.json) within the tolerance
# documented in scripts/verify_reproduction.py.

PY      ?= .venv/bin/python
WORKERS ?= $(shell python3 -c "import os; print(max(1, (os.cpu_count() or 2) - 1))")

.PHONY: install validate-sim tune train val confirm figures report verify test lint \
        reproduce-paper reproduce-paper-fast clean-generated

.venv/bin/python:
	python3 -m venv .venv || uv venv --python 3.12 .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e . --no-deps

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

clean-generated:
	rm -rf models results/raw

reproduce-paper: clean-generated validate-sim tune train val confirm figures report verify test
	@echo "reproduce-paper: done"

reproduce-paper-fast: install
	$(PY) scripts/reproduce_fast.py
