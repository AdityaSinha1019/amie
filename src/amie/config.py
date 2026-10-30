from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Symbol = Literal["EURUSD", "NIFTY50"]
ModelName = Literal["xgboost", "random_forest"]

class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["paper"] = "paper"
    initial_equity: float = Field(default=100_000, gt=0)
    min_train: int = Field(default=100, ge=30)
    test_size: int = Field(default=50, ge=1)
    confidence: float = Field(default=0.60, gt=0.5, lt=1)
    risk_fraction: float = Field(default=0.005, gt=0, le=0.02)
    max_exposure: float = Field(default=1.0, gt=0, le=1)
    max_drawdown: float = Field(default=0.10, gt=0, lt=1)
    eurusd_cost_bps: float = Field(default=1.0, ge=0)
    nifty50_cost_bps: float = Field(default=3.0, ge=0)
    seed: int = 42

    def cost(self, symbol: Symbol) -> float:
        return (self.eurusd_cost_bps if symbol == "EURUSD" else self.nifty50_cost_bps) / 10_000
