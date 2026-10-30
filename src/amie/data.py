from pathlib import Path
from typing import Protocol
import numpy as np
import pandas as pd
from amie.config import Symbol

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

class BarSource(Protocol):
    def load(self, symbol: Symbol) -> pd.DataFrame: ...

def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid input; never sort, deduplicate, fill or infer time zones."""
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError(f"Required columns: {COLUMNS}")
    df = frame[COLUMNS].copy()
    if len(df) < 25:
        raise ValueError("At least 25 completed bars required")
    # Every timestamp must carry an explicit zone, even in CSV input.
    times = [pd.Timestamp(t) for t in df.timestamp]
    if any(pd.isna(t) or t.tzinfo is None for t in times):
        raise ValueError("Timestamps must be timezone-aware")
    df["timestamp"] = pd.to_datetime(times, utc=True)
    if not df.timestamp.is_monotonic_increasing or df.timestamp.duplicated().any():
        raise ValueError("Bars must be strictly chronological and unique")
    for col in COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite bar values")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
        (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bounds")
    return df.reset_index(drop=True)

class CSVSource:
    def __init__(self, directory: Path):
        self.directory = directory

    def load(self, symbol: Symbol) -> pd.DataFrame:
        if symbol not in ("EURUSD", "NIFTY50"):
            raise ValueError("Unsupported symbol")
        return validate_bars(pd.read_csv(self.directory / f"{symbol}.csv"))

def synthetic_bars(symbol: Symbol, count: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily bars, not exchange calendars or historical market data."""
    rng = np.random.default_rng(seed)
    base = 1.08 if symbol == "EURUSD" else 24000.0
    opens = base * np.exp(np.cumsum(rng.normal(0, 0.003, count)))
    closes = opens * np.exp(rng.normal(0, 0.001, count))
    return pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=count, freq="D", tz="UTC"),
        "open": opens, "close": closes, "high": np.maximum(opens, closes)*1.002,
        "low": np.minimum(opens, closes)*0.998, "volume": rng.integers(0, 10000, count)})[COLUMNS]
