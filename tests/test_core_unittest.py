"""Core checks also runnable with unittest when pytest/model libraries are absent."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from pydantic import ValidationError
from amie.config import Settings
from amie.data import CSVSource, synthetic_bars, validate_bars
from amie.features import FEATURES, dataset, features
from amie.risk import signal
from amie.engine import backtest

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.bars = synthetic_bars("EURUSD", 240)

    def test_future_prices_cannot_change_past_features(self):
        changed = self.bars.copy()
        changed.loc[150:, ["open", "high", "low", "close"]] *= 3
        pd.testing.assert_frame_equal(features(changed).iloc[:150], features(self.bars).iloc[:150])

    def test_labels_and_warmup(self):
        data = dataset(self.bars)
        self.assertEqual(data.index[0], 20)
        self.assertEqual(data.index[-1], len(self.bars)-3)
        self.assertFalse(data[FEATURES].isna().any().any())
        for i, row in data.iterrows():
            expected = self.bars.open.iloc[i+2]/self.bars.open.iloc[i+1]-1
            self.assertAlmostEqual(row.forward_return, expected)
            self.assertEqual(row.target, int(expected > 0))
            self.assertEqual(row.label_end, i+2)

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.bars.to_csv(Path(directory)/"EURUSD.csv", index=False)
            loaded = CSVSource(Path(directory)).load("EURUSD")
            self.assertEqual(len(loaded), len(self.bars))
            self.assertTrue(loaded.timestamp.is_monotonic_increasing)

    def test_input_rejections(self):
        for fault in ("duplicates", "reverse", "naive", "nan", "ohlc", "negative_volume"):
            with self.subTest(fault=fault):
                bars = self.bars.copy()
                if fault == "duplicates": bars.loc[1, "timestamp"] = bars.loc[0, "timestamp"]
                if fault == "reverse": bars = bars.iloc[::-1]
                if fault == "naive": bars["timestamp"] = bars.timestamp.dt.tz_localize(None)
                if fault == "nan": bars.loc[0, "close"] = float("nan")
                if fault == "ohlc": bars.loc[0, "high"] = 0.1
                if fault == "negative_volume": bars.loc[0, "volume"] = -1
                with self.assertRaises(ValueError): validate_bars(bars)

    def test_risk_limits(self):
        cfg = Settings()
        self.assertEqual(signal(.5, .01, cfg)["action"], "NO_TRADE")
        self.assertEqual(signal(.9, .01, cfg)["action"], "BUY")
        self.assertEqual(signal(.1, .01, cfg)["action"], "SELL")
        self.assertLessEqual(signal(.9, .00001, cfg)["exposure_fraction"], cfg.max_exposure)
        for p, atr in ((float("nan"), .01), (1.1, .01), (.9, 0)):
            with self.assertRaises(ValueError): signal(p, atr, cfg)

    def test_backtest_long_short_costs_and_execution(self):
        for p, side in ((.9, 1), (.1, -1)):
            rows = pd.DataFrame({"probability_up": [p], "atr_fraction": [.01]}, index=[30])
            cfg = Settings(eurusd_cost_bps=2)
            report = backtest(self.bars, rows, "EURUSD", cfg)
            trade = report["trades"][0]
            ratio = self.bars.open.iloc[32]/self.bars.open.iloc[31]
            self.assertAlmostEqual(trade["notional"], 25000)
            self.assertAlmostEqual(trade["cost"], 25000*.0002*(1+ratio))
            self.assertAlmostEqual(trade["net_pnl"], 25000*side*(ratio-1)-trade["cost"])
            self.assertEqual(trade["entry"], self.bars.open.iloc[31])
            self.assertEqual(trade["exit"], self.bars.open.iloc[32])

    def test_drawdown_halt_and_abstention(self):
        rows = pd.DataFrame({"probability_up": [.9, .9], "atr_fraction": [.001, .001]}, index=[30, 31])
        cfg = Settings(eurusd_cost_bps=1000, max_drawdown=.01)
        report = backtest(self.bars, rows, "EURUSD", cfg)
        self.assertTrue(report["halted"])
        self.assertEqual(report["trade_count"], 1)
        rows["probability_up"] = .5
        report = backtest(self.bars, rows, "EURUSD", cfg)
        self.assertEqual(report["trade_count"], 0)
        self.assertEqual(report["final_equity"], cfg.initial_equity)

    def test_configuration_never_allows_live(self):
        for kwargs in ({"mode": "live"}, {"broker_key": "unused"}, {"max_exposure": 2}):
            with self.assertRaises(ValidationError): Settings(**kwargs)
        Settings.model_validate_json(Path("configs/default.json").read_text())

if __name__ == "__main__":
    unittest.main()
