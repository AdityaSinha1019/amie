#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -c 'from amie.models import classifier; classifier("xgboost", 42)'
.venv/bin/python -m pytest -q
for symbol in EURUSD NIFTY50; do
  for model in random_forest xgboost; do
    .venv/bin/amie evaluate --symbol "$symbol" --model "$model" --output "reports/${symbol}_${model}.json"
  done
done
.venv/bin/python -m pip freeze --exclude-editable > requirements-lock.txt
printf '%s\n' 'Setup and verification complete. Start API: .venv/bin/uvicorn amie.api:app --host 127.0.0.1 --port 8000'
