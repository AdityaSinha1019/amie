import numpy as np
import pandas as pd
from amie.data import validate_bars

FEATURES = ["return_1", "return_5", "ema_gap", "volatility", "atr_fraction", "rsi"]

def features(bars: pd.DataFrame) -> pd.DataFrame:
    """At row t, use only prices known at completed close t. No fitted transforms."""
    df = validate_bars(bars)
    close = df.close
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    tr = pd.concat([df.high-df.low, (df.high-close.shift()).abs(),
                    (df.low-close.shift()).abs()], axis=1).max(axis=1)
    out = pd.DataFrame({"return_1": close.pct_change(), "return_5": close.pct_change(5),
        "ema_gap": close/close.ewm(span=20, adjust=False).mean()-1,
        "volatility": close.pct_change().rolling(20).std(),
        "atr_fraction": tr.rolling(14).mean()/close,
        "rsi": (100 * up/(up+down)).where(up+down != 0, 50)})
    return out.replace([np.inf, -np.inf], np.nan)

def dataset(bars: pd.DataFrame) -> pd.DataFrame:
    bars = validate_bars(bars)
    out = features(bars)
    # Decision close t -> enter open t+1 -> exit open t+2.
    out["forward_return"] = bars.open.shift(-2)/bars.open.shift(-1)-1
    out["label_end"] = np.arange(len(bars)) + 2
    out["target"] = (out.forward_return > 0).astype(int)
    return out.dropna().copy()
