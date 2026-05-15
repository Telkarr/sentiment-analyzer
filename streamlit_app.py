"""Streamlit client for the short-message sentiment API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

EXAMPLES = {
    "Позитив": "Отличный сервис, всё понравилось! Рекомендую",
    "Нейтраль": "Товар доставлен в срок, всё как заказывал",
    "Негатив": "Ужасное качество, полное разочарование",
}

LABEL_EMOJI = {
    "neutral": "Нейтральный",
    "positive": "Позитивный",
    "negative": "Негативный",
}


@st.cache_data(show_spinner=False)
def fetch_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/model_info", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def classify(text: str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
        if r.ok:
            return r.json()
        st.error(f"Ошибка API ({r.status_code}): {r.text}")
    except requests.ConnectionError:
        st.error(f"Не удалось подключиться к API по адресу {API_URL}")
    except Exception as e:
        st.error(f"Ошибка: {e}")
    return None


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
    with st.spinner("Анализирую..."):
        result = classify(input_text.strip())

    if result:
        label_name = result["label_name"]
        proba = result["probabilities"]

        st.subheader("Результат")
        st.metric("Класс", LABEL_EMOJI.get(label_name, label_name))

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
info = fetch_model_info()
if info:
    cfg = info.get("config", {})
    metrics = info.get("metrics_on_test", {})
    f1 = metrics.get("f1_macro", "—")
    vec = cfg.get("vectorizer", "—").upper()
    clf = cfg.get("classifier", "—")
    ngram = cfg.get("ngram_range", [1, 1])
    clf_display = "LogReg" if clf == "logreg" else clf
    st.caption(
        f"Модель: {vec} + биграммы ({ngram[0]},{ngram[1]}) + {clf_display} "
        f"| Macro-F1 (test): {f1:.4f}"
    )
else:
    st.caption("Модель: TF-IDF + биграммы + LogReg | API недоступен")
