.PHONY: lint format check

lint: format check
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