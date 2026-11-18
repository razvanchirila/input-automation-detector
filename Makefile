.DEFAULT_GOAL := help
PYTHON ?= python
IMAGE ?= input-automation-detector:local

.PHONY: help install lint format typecheck test eval lock docker-build docker-run up monitoring down

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install runtime lock plus dev and eval extras (editable)
	$(PYTHON) -m pip install -r requirements.txt -e ".[dev,eval]"

lint: ## ruff lint and format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply ruff fixes and formatting
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck: ## mypy --strict
	$(PYTHON) -m mypy

test: ## pytest with coverage
	$(PYTHON) -m pytest --cov --cov-report=term

eval: ## Full evaluation run, rewrites eval/results.md
	$(PYTHON) eval/evaluate.py

lock: ## Regenerate requirements.txt from pyproject.toml
	$(PYTHON) scripts/lock_requirements.py

docker-build: ## Build the runtime image
	docker build -t $(IMAGE) .

docker-run: ## Run the image on localhost:8000
	docker run --rm -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp --cap-drop ALL $(IMAGE)

up: ## docker compose up (api only)
	docker compose up --build -d

monitoring: ## docker compose up with Prometheus and Grafana
	docker compose --profile monitoring up --build -d

down: ## Stop everything, including the monitoring profile
	docker compose --profile monitoring down
