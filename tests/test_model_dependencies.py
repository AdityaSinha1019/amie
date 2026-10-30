import builtins
import pytest
from fastapi.testclient import TestClient
from amie.api import app
from amie.data import synthetic_bars
from amie.models import ModelUnavailableError, classifier

@pytest.fixture
def missing_xgboost(monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ValueError("Library not loaded: libomp.dylib")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)

def test_random_forest_does_not_import_xgboost(missing_xgboost):
    assert classifier("random_forest", 42).n_estimators == 100

def test_missing_runtime_has_actionable_error(missing_xgboost):
    with pytest.raises(ModelUnavailableError, match="brew install libomp"):
        classifier("xgboost", 42)

@pytest.mark.parametrize("endpoint", ["/evaluate", "/signal"])
def test_api_dependency_failure_is_service_unavailable(missing_xgboost, endpoint):
    bars = synthetic_bars("EURUSD", 160)
    payload = {"symbol": "EURUSD", "model": "xgboost", "bars": [
        {**row, "timestamp": str(row["timestamp"])} for row in bars.to_dict("records")]}
    response = TestClient(app).post(endpoint, json=payload)
    assert response.status_code == 503
    assert "brew install libomp" in response.json()["detail"]
