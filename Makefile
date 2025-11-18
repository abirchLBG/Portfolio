.PHONY: lint format check

lint: format check lint_test
ruff: ruff_format_check

format:
	@echo "Running Ruff format check..."
	@uvx ruff format --check

check:
	@echo "Running Ruff lint..."
	@uvx ruff check 

ruff_format_check:
	@echo "Running ruff format and check"
	@uvx ruff format
	@uvx ruff check --fix

lint_test:
	@echo "Running tests"
	@uv run pytest -qq .

test:
	@echo "Running tests"
# 	@uv run pytest -n=auto .
	@uv run pytest .

cov:
	@echo "Checking test coverage"
	@uv run pytest --cov=.

build:
	@echo "Building"
	@docker compose build

run:
	@echo "Deploying k8s"
	@docker compose build
	@docker compose up --scale worker=1