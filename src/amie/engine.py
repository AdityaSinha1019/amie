import pandas as pd
from amie.config import Settings, Symbol, ModelName
from amie.data import validate_bars
from amie.features import FEATURES, dataset, features
from amie.risk import signal

def walk_forward(bars: pd.DataFrame, name: ModelName, cfg: Settings) -> tuple[pd.DataFrame, list]:
    from sklearn.metrics import accuracy_score, log_loss
    from amie.models import classifier

    data = dataset(bars)
    predictions, folds = [], []
    # Gap of two decision rows: training outcomes strictly precede test decisions.
    for start in range(cfg.min_train+2, len(data), cfg.test_size):
        test = data.iloc[start:start+cfg.test_size]
        train = data.iloc[:start-2]
        train = train[train.label_end < test.index[0]]
        if train.target.nunique() != 2:
            raise ValueError("Training fold must contain both direction classes")
        model = classifier(name, cfg.seed)
        model.fit(train[FEATURES], train.target)
        proba = model.predict_proba(test[FEATURES])[:, 1]
        result = test.copy()
        result["probability_up"] = proba
        predictions.append(result)
        folds.append({"train_rows": len(train), "train_last_decision": int(train.index[-1]),
            "train_last_label_end": int(train.label_end.max()), "test_first_decision": int(test.index[0]),
            "test_rows": len(test), "accuracy": float(accuracy_score(test.target, proba >= .5)),
            "log_loss": float(log_loss(test.target, proba, labels=[0, 1]))})
    if not predictions:
        raise ValueError("Not enough bars for a walk-forward fold")
    return pd.concat(predictions), folds

def backtest(bars: pd.DataFrame, predictions: pd.DataFrame, symbol: Symbol, cfg: Settings) -> dict:
    equity = peak = cfg.initial_equity
    trades, curve = [], [equity]
    halted = False
    last_exit = -1
    for i, row in predictions.iterrows():
        if halted or i+1 < last_exit:
            continue
        decision = signal(float(row.probability_up), float(row.atr_fraction), cfg)
        if not decision["side"]:
            continue
        entry, exit_price = float(bars.open.iloc[i+1]), float(bars.open.iloc[i+2])
        notional = equity * decision["exposure_fraction"]
        gross = notional * decision["side"] * (exit_price/entry-1)
        # Per-side spread + slippage + fees, charged on actual entry/exit notional.
        cost = notional * cfg.cost(symbol) * (1+exit_price/entry)
        equity += gross-cost
        peak = max(peak, equity)
        curve.append(equity)
        halted = equity <= peak*(1-cfg.max_drawdown)
        last_exit = i+2
        trades.append({"decision_time": str(bars.timestamp.iloc[i]), "entry_time": str(bars.timestamp.iloc[i+1]),
            "exit_time": str(bars.timestamp.iloc[i+2]), "side": decision["action"],
            "entry": entry, "exit": exit_price, "notional": notional,
            "gross_pnl": gross, "cost": cost, "net_pnl": gross-cost, "equity": equity})
    values = pd.Series(curve)
    return {"initial_equity": cfg.initial_equity, "final_equity": equity,
        "return_fraction": equity/cfg.initial_equity-1,
        "max_drawdown": float((1-values/values.cummax()).max()),
        "halted": halted, "trade_count": len(trades), "trades": trades}

def evaluate(bars: pd.DataFrame, symbol: Symbol, name: ModelName, cfg: Settings) -> dict:
    bars = validate_bars(bars)
    predictions, folds = walk_forward(bars, name, cfg)
    return {"symbol": symbol, "model": name, "mode": "paper", "settings": cfg.model_dump(),
        "folds": folds, "backtest": backtest(bars, predictions, symbol, cfg)}

def latest_signal(bars: pd.DataFrame, name: ModelName, cfg: Settings) -> dict:
    from amie.models import classifier

    bars = validate_bars(bars)
    data = dataset(bars)
    # Final close known, but train only labels ending strictly before this decision.
    train = data[data.label_end < len(bars)-1]
    if len(train) < cfg.min_train or train.target.nunique() != 2:
        raise ValueError("Insufficient training history or direction classes")
    model = classifier(name, cfg.seed)
    model.fit(train[FEATURES], train.target)
    latest = features(bars).iloc[[-1]]
    p = float(model.predict_proba(latest[FEATURES])[0, 1])
    return {**signal(p, float(latest.atr_fraction.iloc[0]), cfg),
        "decision_time": str(bars.timestamp.iloc[-1]), "model": name,
        "execution": "hypothetical next open; no orders submitted"}
