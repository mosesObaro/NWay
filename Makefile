VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help venv install test tick dry-run ingest train backtest report clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

venv:           ## create the virtualenv
	python3 -m venv $(VENV) --system-site-packages

install: venv   ## install dependencies
	$(PIP) install -q -r requirements.txt

test:           ## run the full test suite
	$(PY) -m pytest -q

test-leakage:   ## run only the temporal-leakage suite
	$(PY) -m pytest tests/leakage -q

init-db:        ## create the database schema
	$(PY) -m nway.cli init-db

ingest:         ## ingest all configured sources
	$(PY) -m nway.cli ingest

train:          ## train the active models
	$(PY) -m nway.cli train

backtest:       ## run the walk-forward backtest
	$(PY) -m nway.cli backtest

tick:           ## one scheduler tick (may send email)
	$(PY) -m nway.cli tick

dry-run:        ## one scheduler tick, decide but send nothing
	$(PY) -m nway.cli tick --dry-run

report:         ## evaluation report
	$(PY) -m nway.cli report

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
