install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

check:
	ruff check .
	pytest -q
