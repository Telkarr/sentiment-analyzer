"""Streamlit app for short-message sentiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import nltk
import joblib
import numpy as np
import streamlit as st

nltk.download("stopwords", quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_NAMES = ["neutral", "positive", "negative"]

EXAMPLES = {
    "Позитив": "Отличный сервис, всё понравилось! Рекомендую",
    "Нейтраль": "Всё неплохо, но ожидал большего",
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
    placeholder="Напишите текст или выберите пример в текстовом поле...",
)

if st.button("Классифицировать", type="primary"):
    if not input_text.strip():
        st.warning("Введите текст для анализа")
        st.stop()
    result = classify(input_text.strip())
    label_name = result["label_name"]
    proba = result["probabilities"]

    LABEL_STYLE = {
        "positive": ("Позитивный", "rgba(34, 197, 94, 0.15)", "#16a34a"),
        "neutral":  ("Нейтральный", "rgba(156, 163, 175, 0.15)", "#6b7280"),
        "negative": ("Негативный",  "rgba(239, 68, 68, 0.15)",  "#dc2626"),
    }
    display, bg, border = LABEL_STYLE[label_name]
    st.markdown(
        f"""<div style="background:{bg};border:1.5px solid {border};
        border-radius:10px;padding:16px 24px;display:inline-block;
        font-size:1.3rem;font-weight:600;color:var(--text-color)">{display}</div>""",
        unsafe_allow_html=True,
    )
    st.write("")

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
