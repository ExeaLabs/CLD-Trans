.PHONY: setup test lint typecheck ci

setup:
	bash scripts/setup_env.sh

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy modules models losses data engine analysis

ci: lint typecheck test
