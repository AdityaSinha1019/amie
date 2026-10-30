from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from amie.config import Settings, Symbol, ModelName
from amie.engine import evaluate, latest_signal
from amie.models import ModelUnavailableError

app = FastAPI(title="AMIE paper research API", version="0.1.0")

class Bar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0

class ResearchRequest(BaseModel):
    symbol: Symbol
    model: ModelName = "random_forest"
    bars: list[Bar] = Field(min_length=25, max_length=10000)
    settings: Settings = Field(default_factory=Settings)

@app.get("/health")
def health():
    return {"status": "ok", "mode": "paper", "live_trading": False}

@app.post("/evaluate")
def evaluation(request: ResearchRequest):
    try:
        return evaluate(pd.DataFrame([b.model_dump() for b in request.bars]),
                        request.symbol, request.model, request.settings)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/signal")
def prediction(request: ResearchRequest):
    try:
        return {"symbol": request.symbol, **latest_signal(
            pd.DataFrame([b.model_dump() for b in request.bars]), request.model, request.settings)}
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
