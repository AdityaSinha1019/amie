import math
from amie.config import Settings

def signal(probability_up: float, atr_fraction: float, cfg: Settings) -> dict:
    if not all(math.isfinite(x) for x in (probability_up, atr_fraction)) or not 0 <= probability_up <= 1 or atr_fraction <= 0:
        raise ValueError("Invalid probability or ATR")
    side = 1 if probability_up >= cfg.confidence else -1 if probability_up <= 1-cfg.confidence else 0
    # ATR is a sizing proxy, not a guaranteed maximum loss or an executed stop.
    fraction = min(cfg.max_exposure, cfg.risk_fraction/(2*atr_fraction)) if side else 0.0
    return {"action": {1: "BUY", -1: "SELL", 0: "NO_TRADE"}[side],
            "side": side, "exposure_fraction": fraction, "probability_up": probability_up,
            "mode": "paper"}
