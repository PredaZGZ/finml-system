.PHONY: install test lint clean format

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf dist build .pytest_cache .ruff_cache src/*.egg-info __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
