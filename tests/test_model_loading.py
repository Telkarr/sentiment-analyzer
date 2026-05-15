"""Smoke test: load the trained model and run a single prediction."""

from __future__ import annotations

from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")
    result = model.predict(["Отличный сервис, всё понравилось!"])
    assert len(result) == 1
    assert int(result[0]) in {0, 1, 2}
