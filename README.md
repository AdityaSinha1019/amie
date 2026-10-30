# AMIE — Adaptive Market Intelligence Engine

Python-first, **paper-trading-only** research MVP for EUR/USD (`EURUSD`) and NIFTY 50 (`NIFTY50`). Separate XGBoost and Random Forest binary direction classifiers are fitted per symbol and fold. No broker integration, credentials, live mode, order endpoint, or network data fetching exists.

## Setup

On macOS, XGBoost requires the OpenMP runtime: run `brew install libomp` before setup. Random Forest loads independently of XGBoost. Missing XGBoost runtime dependencies return HTTP 503 from research endpoints.

One-command installation, full tests, all four evaluations and environment snapshot: `sh scripts/setup.sh` (requires internet access). Current verification status is in [reports/VALIDATION.md](reports/VALIDATION.md).

Use Python 3.11–3.14, in this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
# Sample CSVs are included; generate fresh fixtures only in an empty directory:
# amie sample --data-dir /tmp/amie-samples
amie evaluate --symbol EURUSD --model random_forest --output reports/eurusd_rf.json
amie evaluate --symbol EURUSD --model xgboost --output reports/eurusd_xgb.json
amie evaluate --symbol NIFTY50 --model random_forest --output reports/nifty50_rf.json
amie evaluate --symbol NIFTY50 --model xgboost --output reports/nifty50_xgb.json
uvicorn amie.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation: http://127.0.0.1:8000/docs. Sample generation refuses to overwrite existing CSVs. Committed samples are synthetic; skip `amie sample` if they already exist. Dependency ranges are in `pyproject.toml`. After a successful installation and test run, record your environment with `python -m pip freeze > requirements-lock.txt`; no unverified lockfile is supplied.

## Data contract

`CSVSource(directory).load(symbol)` implements the `BarSource` protocol. Supply `data/EURUSD.csv` and `data/NIFTY50.csv`, or pass `--data-dir`. A provider adapter should implement the same `load` method without exposing credentials to this package.

Required columns:

```csv
timestamp,open,high,low,close,volume
2025-01-01T00:00:00Z,1.0800,1.0820,1.0790,1.0810,0
```

At least 25 bars are required for validation, and at least `min_train + 25` for an evaluation fold. Timestamp means **completed bar close**. Bars must be strictly increasing, unique and explicitly timezone-aware; storage normalizes to UTC. Prices must be finite, positive and within valid OHLC bounds; volume must be finite and nonnegative. No sorting, backfilling or duplicate removal hides input problems. The caller must exclude incomplete bars and establish provenance, market calendars, bar frequency and quote conventions. Zero volume is accepted for index/FX feeds. Features currently do not use volume.

Synthetic samples use daily UTC timestamps including weekends. They are pipeline fixtures, not realistic session data, intraday histories or evidence of profitability. Never combine instruments in one CSV. NIFTY 50 is an index, so this MVP simulates fractional index exposure in abstract quote-currency units; it does not model an executable futures, option or ETF contract. Each symbol has its own account; cross-currency portfolios are not aggregated.

## Chronology and models

1. Features at close `t`: 1/5-bar returns, causal EMA gap, 20-bar volatility, 14-bar ATR fraction and RSI. Rolling warm-up rows are dropped; no future fill or fitted preprocessing is used.
2. Signal is available after completed close `t`. Hypothetical entry is open `t+1`, exit open `t+2`. Label is 1 when that open-to-open return is positive, otherwise 0. The last two rows have unknown labels and are excluded from training/evaluation.
3. Expanding walk-forward windows use `min_train` and `test_size`. Two decision rows are purged, and training label-end indices must strictly precede the first test decision. Models fit only the training fold. Both classes must exist or evaluation fails explicitly. Test tails are retained.
4. Each classifier produces its own out-of-sample predictions, fold accuracy and log loss. No ensemble, random split, tuning on test rows or calibration is applied. Probability outputs are uncalibrated.
5. A latest-signal request trains a fresh model on resolved historical labels and uses the final completed bar's features. It is an advisory paper signal, independent of backtest account state. It does not persist a model or submit an order.

Design reference: [scikit-learn's chronological splitting and gap documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). AMIE implements explicit index-based folds so label-end assertions can be audited.

## Costs, risk and execution

`configs/default.json` supplies validated settings; use `--config PATH` to override. Live mode and unknown configuration fields are rejected. Per-side costs default to **illustrative** 1 bp EUR/USD and 3 bp NIFTY 50; these are assumptions, not researched broker tariffs. Each cost bundles spread, fees and slippage. Round-trip cost is `entry_notional * per_side_rate * (1 + exit / entry)` and applies to buys and sells alike. Labels predict gross direction; costs affect realized backtests, not the classification target.

Signals buy above the confidence threshold, sell below its complement, otherwise abstain. Exposure is `min(max_exposure, risk_fraction / (2 * ATR_fraction))`; default maximum gross exposure is 100% with no leverage. ATR is a volatility sizing proxy, **not a stop loss or maximum-loss guarantee**. Positions exit at the next scheduled open; gap losses can exceed the sizing budget. This MVP has no stop/target, financing, borrow, margin, partial fills, lot rounding, taxes or market impact model. Same-open exit/re-entry is permitted; costs are charged for each leg. Entries cannot overlap existing positions.

Equity compounds net P&L. A peak-to-equity drawdown breach permanently halts new entries for that backtest, after closing the current trade. Drawdown is measured at exits only, so intratrade drawdown can be larger. Account equity and performance reports use independent quote-currency units. Do not interpret synthetic performance as a trading recommendation.

## API

- `GET /health`: reports paper mode and disabled live trading.
- `POST /evaluate`: walk-forward model metrics and cost-aware paper trade ledger.
- `POST /signal`: latest paper signal, exposure fraction and decision timestamp.

Both POST endpoints accept `{ "symbol": "EURUSD", "model": "random_forest", "bars": [...], "settings": {...} }`. Each bar uses the CSV column names; timestamps must include a zone. Model can also be `xgboost`. Settings can be omitted. Inputs are limited to 10,000 bars. These synchronous research endpoints retrain models on every call; serve locally. Authentication, job queues and durable account state are outside the MVP.

## Repository and next steps

- `src/amie/config.py`: paper-only validated settings and symbol costs.
- `src/amie/data.py`: ingestion protocol, CSV validation, synthetic fixtures.
- `src/amie/features.py`: causal features and future labels with outcome indices.
- `src/amie/models.py`: separate model factories.
- `src/amie/engine.py`: purged walk-forward fitting, ledger and latest prediction.
- `src/amie/risk.py`: confidence filter and capped exposure sizing.
- `src/amie/api.py`, `cli.py`: local service and reproducible evaluation commands.
- `tests/`: leakage, label timing, model/fold, costs, risk, ingestion and API checks.
- `data/`: clearly synthetic samples; `reports/`: generated evaluation JSONs.

Next: source licensed historical bars; validate sessions and missing bars; choose the actual NIFTY proxy and contract rules; replace illustrative costs with documented assumptions; add untouched final holdout and inner chronological tuning/calibration; benchmark against simple direction/no-trade baselines; add intrabar risk/execution rules, persistent paper accounts, model artifacts and drift monitoring. Intraday support needs explicit session flattening and calendar checks before its results are meaningful.

## Collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for partner setup, feature branches, reviews, and historical-data handling. Real market downloads, generated evaluation JSONs, credentials and trained artifacts are excluded from Git; the two committed CSVs are synthetic fixtures.
