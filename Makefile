.PHONY: lint format check

lint: format check

format:
	@echo "Running Ruff format check..."
	@uvx ruff format --check

check:
	@echo "Running Ruff lint..."
	@uvx ruff check 
