# Verification status — 2026-10-30

OpenMP is installed and XGBoost loads successfully in the project's `.venv`.

- Full pytest suite: **26 passed**, plus **6 subtests passed**, no deselections or skips.
- One upstream Starlette/httpx deprecation warning remains; it does not affect the passing API tests.
- All four CLI walk-forward evaluations completed successfully:

| Symbol | Classifier | Synthetic trades | Report |
| --- | --- | ---: | --- |
| EURUSD | Random Forest | 186 | EURUSD_random_forest.json |
| EURUSD | XGBoost | 253 | EURUSD_xgboost.json |
| NIFTY50 | Random Forest | 149 | NIFTY50_random_forest.json |
| NIFTY50 | XGBoost | 258 | NIFTY50_xgboost.json |

Each JSON report contains configuration, chronological fold metrics and a transaction-cost-aware paper ledger. Installed Python package versions are recorded in `requirements-lock.txt`; the system OpenMP dependency is separate.

All supplied market data is synthetic. These results validate pipeline operation, not trading performance. No live trading or broker connection is enabled.
