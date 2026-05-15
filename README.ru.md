# Sentiment Analyzer

[English](README.md) | **Русский**

Веб-приложение для анализа тональности коротких русскоязычных текстов (до 280 символов). Классифицирует сообщения на три класса: **позитив / нейтраль / негатив**.

Модель: TF-IDF + биграммы + логистическая регрессия, macro-F1 = **0.7318**.

---

## Быстрый старт

### Установка

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### Запуск API

```bash
uvicorn src.api:app --port 8000
```

### Запуск Streamlit

```bash
streamlit run streamlit_app.py
```

Открывается по адресу [http://localhost:8501](http://localhost:8501).

По умолчанию приложение обращается к API на `http://localhost:8000`. Адрес можно переопределить через переменную окружения:

```bash
API_URL=http://your-api-host:8000 streamlit run streamlit_app.py
```

---

## API

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Проверка доступности |
| GET | `/model_info` | Метаданные модели |
| POST | `/predict` | Классификация одного текста |
| POST | `/predict_batch` | Классификация до 100 текстов |

Пример:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Отличный сервис, всё понравилось!"}'
```

```json
{"label": 1, "label_name": "positive", "probabilities": {"neutral": 0.003, "positive": 0.995, "negative": 0.001}}
```

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
  api.py               # FastAPI-сервис
  preprocessing.py     # очистка и лемматизация текста
models/
  best_model.joblib    # обученная модель (sklearn Pipeline)
  model_metadata.json  # метрики и конфигурация
tests/                 # юнит и интеграционные тесты
requirements.txt
```

---

## Лицензия

MIT
