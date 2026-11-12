.PHONY: install test lint api frontend

install:
	python -m pip install -e .[dev]

test:
	python -m pytest

lint:
	python -m ruff check src tests

api:
	uvicorn campus_forecast.api.main:app --reload

frontend:
	cd frontend && npm install && npm run dev
