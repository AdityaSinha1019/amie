# Working together on AMIE

## First setup

Clone the shared repository, open its directory, then run:

```sh
# macOS prerequisite for XGBoost:
brew install libomp
sh scripts/setup.sh
```

On Linux, skip the Homebrew command. Setup creates `.venv`, installs dependencies, runs tests, and generates synthetic paper evaluations. The README describes individual commands.

## Working on a change

```sh
git switch main
git pull --ff-only
git switch -c feature/describe-your-change
```

Make a focused change, run `.venv/bin/python -m pytest -q`, then commit and push the branch. Open a pull request for your partner to review before merging. Avoid committing directly to `main` when collaborating.

## Historical data and training

Keep original downloads under `data/raw/` and normalized files under `data/historical/`:

```text
data/historical/EURUSD.csv
data/historical/NIFTY50.csv
```

These folders are ignored by Git. Share data separately under your provider's redistribution terms. Document the provider, download date, timezone, timeframe and coverage with the shared data. The committed CSVs in `data/` are synthetic fixtures.

Train and evaluate using chronological walk-forward splits:

```sh
.venv/bin/amie evaluate --data-dir data/historical --symbol EURUSD --model xgboost --output reports/eurusd_historical_xgb.json
```

Use `--symbol NIFTY50` or `--model random_forest` for the other combinations. Evaluation JSONs and model artifacts stay local. Never commit credentials, `.env` files or virtual environments. Keep execution paper-only and preserve feature/label chronology when changing the pipeline.
