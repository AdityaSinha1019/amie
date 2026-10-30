import argparse
import json
from pathlib import Path
from amie.config import Settings
from amie.data import CSVSource, synthetic_bars
from amie.engine import evaluate

def main():
    parser = argparse.ArgumentParser(description="AMIE paper-only research")
    parser.add_argument("command", choices=["sample", "evaluate"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbol", choices=["EURUSD", "NIFTY50"], default="EURUSD")
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="random_forest")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    if args.command == "sample":
        args.data_dir.mkdir(parents=True, exist_ok=True)
        for symbol in ("EURUSD", "NIFTY50"):
            path = args.data_dir / f"{symbol}.csv"
            if path.exists():
                raise SystemExit(f"Refusing to overwrite {path}")
            synthetic_bars(symbol, seed=42 if symbol == "EURUSD" else 43).to_csv(path, index=False)
        print("Created SYNTHETIC daily samples for both symbols")
        return
    cfg = Settings.model_validate_json(args.config.read_text())
    report = evaluate(CSVSource(args.data_dir).load(args.symbol), args.symbol, args.model, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Paper evaluation saved: {args.output}; trades={report['backtest']['trade_count']}")
