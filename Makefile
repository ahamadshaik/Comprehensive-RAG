.PHONY: setup ingest eval serve test

setup:
	python -m venv .venv
	@echo "Activate: source .venv/bin/activate (Unix) or .venv\\Scripts\\activate (Windows)"
	pip install -r requirements.txt

ingest:
	python -m app.ingest

eval:
	python eval/evaluate.py

serve:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/ -q
