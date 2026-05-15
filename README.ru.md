# Sentiment Analyzer

[English](README.md) | **Русский**

Веб-приложение для анализа тональности коротких русскоязычных текстов (до 5000 символов). Классифицирует сообщения на три класса: **позитив / нейтраль / негатив**.

Модель: TF-IDF + биграммы + логистическая регрессия, macro-F1 = **0.7318**.

---

## Быстрый старт

### Установка

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### Запуск

```bash
streamlit run streamlit_app.py
```

Открывается по адресу [http://localhost:8501](http://localhost:8501).

---

## Тесты

```bash
pytest tests/ -v
```

---

## Структура проекта

```
streamlit_app.py       # Streamlit-интерфейс
src/
  preprocessing.py     # очистка и лемматизация текста
models/
  best_model.joblib    # обученная модель (sklearn Pipeline)
  model_metadata.json  # метрики и конфигурация
tests/                 # тесты
requirements.txt
```

---

## Лицензия

MIT
