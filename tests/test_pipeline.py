import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from amie.api import app
from amie.config import Settings
from amie.data import synthetic_bars, validate_bars, CSVSource
from amie.features import features, dataset, FEATURES
from amie.engine import walk_forward, backtest, latest_signal
from amie.risk import signal

@pytest.fixture
def bars():
    return synthetic_bars("EURUSD", 240)

def test_future_changes_do_not_change_past_features(bars):
    altered = bars.copy()
    altered.loc[150:, ["open", "high", "low", "close"]] *= 2
    pd.testing.assert_frame_equal(features(bars).iloc[:150], features(altered).iloc[:150])

def test_labels_match_execution_and_tail_is_removed(bars):
    data = dataset(bars)
    assert data.index[-1] == len(bars)-3
    i = data.index[0]
    assert data.loc[i, "forward_return"] == pytest.approx(bars.open.iloc[i+2]/bars.open.iloc[i+1]-1)
    assert not data[FEATURES].isna().any().any()

@pytest.mark.parametrize("name", ["random_forest", "xgboost"])
def test_purged_folds_and_models(bars, name):
    cfg = Settings(min_train=50, test_size=60)
    predictions, folds = walk_forward(bars, name, cfg)
    assert predictions.index.is_unique
    assert all(f["train_last_label_end"] < f["test_first_decision"] for f in folds)
    assert predictions.probability_up.between(0, 1).all()
    assert sum(f["test_rows"] for f in folds) == len(predictions)

@pytest.mark.parametrize("bad", ["duplicate", "reverse", "naive", "nan", "ohlc"])
def test_invalid_bars_rejected(bars, bad):
    if bad == "duplicate": bars.loc[1, "timestamp"] = bars.loc[0, "timestamp"]
    if bad == "reverse": bars = bars.iloc[::-1]
    if bad == "naive": bars["timestamp"] = bars.timestamp.dt.tz_localize(None)
    if bad == "nan": bars.loc[0, "close"] = np.nan
    if bad == "ohlc": bars.loc[0, "high"] = 0.1
    with pytest.raises(ValueError): validate_bars(bars)

def test_cost_math_and_next_open_execution(bars):
    rows = pd.DataFrame({"probability_up": [0.9], "atr_fraction": [0.01]}, index=[30])
    cfg = Settings(eurusd_cost_bps=2, max_exposure=0.5)
    report = backtest(bars, rows, "EURUSD", cfg)
    trade = report["trades"][0]
    ratio = bars.open.iloc[32]/bars.open.iloc[31]
    assert trade["entry"] == bars.open.iloc[31]
    assert trade["notional"] == 25000
    assert trade["cost"] == pytest.approx(25000*0.0002*(1+ratio))
    assert trade["net_pnl"] == pytest.approx(25000*(ratio-1)-trade["cost"])
    zero = backtest(bars, rows, "EURUSD", Settings(eurusd_cost_bps=0))
    assert zero["final_equity"] > report["final_equity"]

def test_drawdown_halts_further_entries(bars):
    rows = pd.DataFrame({"probability_up": [0.9, 0.9], "atr_fraction": [0.001, 0.001]}, index=[30, 31])
    report = backtest(bars, rows, "EURUSD", Settings(eurusd_cost_bps=1000, max_drawdown=0.01))
    assert report["halted"]
    assert report["trade_count"] == 1

def test_risk_and_paper_only_configuration():
    cfg = Settings()
    assert signal(.5, .01, cfg)["action"] == "NO_TRADE"
    assert signal(.9, .001, cfg)["exposure_fraction"] <= 1
    assert signal(.1, .01, cfg)["action"] == "SELL"
    with pytest.raises(ValidationError): Settings(mode="live")

def test_csv_round_trip_and_latest_signal(bars, tmp_path):
    bars.to_csv(tmp_path / "EURUSD.csv", index=False)
    loaded = CSVSource(tmp_path).load("EURUSD")
    result = latest_signal(loaded, "random_forest", Settings())
    assert result["mode"] == "paper"
    assert result["decision_time"] == str(bars.timestamp.iloc[-1])

def test_api(bars):
    client = TestClient(app)
    assert client.get("/health").json()["live_trading"] is False
    payload = {"symbol": "NIFTY50", "bars": []}
    assert client.post("/evaluate", json=payload).status_code == 422
    payload["bars"] = [{**r, "timestamp": str(r["timestamp"])} for r in bars.to_dict("records")]
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["folds"]
    response = client.post("/signal", json=payload)
    assert response.status_code == 200
    assert response.json()["symbol"] == "NIFTY50"
    assert client.post("/orders", json={}).status_code == 404
    payload["settings"] = {"mode": "live"}
    assert client.post("/evaluate", json=payload).status_code == 422
