.PHONY: install test test-all lint olmo
install:
	uv sync --extra dev
test:
	uv run pytest
test-all:
	uv run pytest -m ""
lint:
	uv run ruff check .
olmo:
	uv sync --extra dev --extra olmo && uv run python scripts/olmo_measure.py all
