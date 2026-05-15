"""Streamlit app for short-message sentiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_NAMES = ["neutral", "positive", "negative"]

LABEL_DISPLAY = {
    "neutral": "Нейтральный",
    "positive": "Позитивный",
    "negative": "Негативный",
}

EXAMPLES = {
    "Позитив": "Отличный сервис, всё понравилось! Рекомендую",
    "Нейтраль": "Товар доставлен в срок, всё как заказывал",
    "Негатив": "Ужасное качество, полное разочарование",
}


@st.cache_resource(show_spinner="Загружаю модель...")
def load_model():
    return joblib.load(MODELS_DIR / "best_model.joblib")


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))


def get_probabilities(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    scores = estimator.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def classify(text: str) -> dict:
    model = load_model()
    label = int(model.predict([text])[0])
    proba = get_probabilities(model, [text])[0]
    return {
        "label_name": LABEL_NAMES[label],
        "probabilities": {
            "positive": float(round(proba[1], 4)),
            "neutral": float(round(proba[0], 4)),
            "negative": float(round(proba[2], 4)),
        },
    }


st.title("Анализ тональности коротких сообщений")

with st.sidebar:
    st.header("Примеры")
    for label, example_text in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["input_text"] = example_text

input_text = st.text_area(
    "Введите текст для анализа",
    value=st.session_state.get("input_text", ""),
    height=120,
    max_chars=5000,
    placeholder="Напишите текст или выберите пример в сайдбаре...",
)

if st.button("Классифицировать", type="primary", disabled=not input_text.strip()):
    result = classify(input_text.strip())
    label_name = result["label_name"]
    proba = result["probabilities"]

    st.subheader("Результат")
    st.metric("Класс", LABEL_DISPLAY.get(label_name, label_name))

    st.subheader("Вероятности")
    for cls_key, cls_display in [
        ("positive", "Позитивный"),
        ("neutral", "Нейтральный"),
        ("negative", "Негативный"),
    ]:
        val = proba[cls_key]
        st.write(f"**{cls_display}** — {val:.1%}")
        st.progress(val)

st.divider()
metadata = load_metadata()
cfg = metadata.get("config", {})
metrics = metadata.get("metrics_on_test", {})
f1 = metrics.get("f1_macro", "—")
vec = cfg.get("vectorizer", "—").upper()
clf = cfg.get("classifier", "—")
ngram = cfg.get("ngram_range", [1, 1])
clf_display = "LogReg" if clf == "logreg" else clf
st.caption(
    f"Модель: {vec} + биграммы ({ngram[0]},{ngram[1]}) + {clf_display} "
    f"| Macro-F1 (test): {f1:.4f}"
)
