"""FastAPI service for short-message sentiment prediction."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_NAMES = ["neutral", "positive", "negative"]


def get_probabilities(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    scores = estimator.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODELS_DIR / "best_model.joblib")
    app.state.metadata = json.loads(
        (MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8")
    )
    yield


app = FastAPI(title="Sentiment API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 5000:
            raise ValueError("text too long (max 5000 chars)")
        return v


class PredictBatchRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("texts must not be empty")
        if len(v) > 100:
            raise ValueError("too many texts (max 100)")
        result = []
        for i, t in enumerate(v):
            t = t.strip()
            if not t:
                raise ValueError(f"texts[{i}] must not be empty")
            if len(t) > 5000:
                raise ValueError(f"texts[{i}] too long (max 5000 chars)")
            result.append(t)
        return result


class ProbabilityDict(BaseModel):
    neutral: float
    positive: float
    negative: float


class PredictResponse(BaseModel):
    label: int
    label_name: str
    probabilities: ProbabilityDict


def _make_response(label: int, proba_row: np.ndarray) -> PredictResponse:
    return PredictResponse(
        label=int(label),
        label_name=LABEL_NAMES[label],
        probabilities=ProbabilityDict(
            neutral=float(round(proba_row[0], 4)),
            positive=float(round(proba_row[1], 4)),
            negative=float(round(proba_row[2], 4)),
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model_info")
def model_info(request: Request):
    return request.app.state.metadata


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    model = request.app.state.model
    labels = model.predict([body.text])
    proba = get_probabilities(model, [body.text])
    return _make_response(int(labels[0]), proba[0])


@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch(body: PredictBatchRequest, request: Request):
    model = request.app.state.model
    labels = model.predict(body.texts)
    proba = get_probabilities(model, body.texts)
    return [_make_response(int(lbl), p) for lbl, p in zip(labels, proba)]
