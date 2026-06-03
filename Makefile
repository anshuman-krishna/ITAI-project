.PHONY: help up down logs build ps clean test lint analyze benchmark train report-assets pipeline

# python on PATH by default; override with `make PYTHON=.venv/bin/python ...` to use a venv.
PYTHON ?= python
MLPATH := PYTHONPATH=packages/ml-core

help:
	@echo "stack:"
	@echo "  up            start the full stack (detached)"
	@echo "  down          stop the stack"
	@echo "  logs          tail logs for all services"
	@echo "  build         rebuild all images"
	@echo "  ps            list running services"
	@echo "  clean         stop and remove volumes"
	@echo "ml workflow (set PYTHON to your interpreter, e.g. .venv/bin/python):"
	@echo "  test          run ml-core + api tests"
	@echo "  lint          ruff check the python code"
	@echo "  analyze       profile + validate the dataset -> reports/dataset"
	@echo "  benchmark     run detection + prompt_id benchmarks -> reports/benchmarks"
	@echo "  train         train + persist the detector, write explainability"
	@echo "  report-assets assemble docs/report-assets from reports/"
	@echo "  pipeline      analyze + benchmark + train + report-assets"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

ps:
	docker compose ps

clean:
	docker compose down -v

test:
	cd packages/ml-core && $(PYTHON) -m pytest -q
	cd apps/api && PYTHONPATH=../../packages/ml-core $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check packages/ml-core apps/api scripts

analyze:
	$(MLPATH) $(PYTHON) scripts/analyze_dataset.py

benchmark:
	$(MLPATH) $(PYTHON) scripts/run_benchmark_suite.py --label-col generated --name detection \
		--title "AI-text detection benchmark (primary objective)" \
		--note "Key finding: the dataset carries only 3 AI-generated essays out of 1,378, so every classifier collapses to the majority class 'human'. Accuracy near 0.998 is the majority baseline, not detection skill (AI-class recall is 0)."
	$(MLPATH) $(PYTHON) scripts/run_benchmark_suite.py --label-col prompt_id --name prompt_id \
		--title "System-validation benchmark — prompt classification (auxiliary)" \
		--note "Not the AI-detection task. A balanced, real-text binary classification used to confirm the benchmarking pipeline produces meaningful, differentiated results."

train:
	$(MLPATH) $(PYTHON) scripts/train_model.py

report-assets:
	$(PYTHON) scripts/build_report_assets.py

pipeline: analyze benchmark train report-assets
