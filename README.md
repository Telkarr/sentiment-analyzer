# Sentiment Analyzer

**English** | [Русский](README.ru.md)

Web application for sentiment analysis of short Russian-language texts (up to 280 characters). Classifies messages into three categories: **positive / neutral / negative**.

Model: TF-IDF + bigrams + logistic regression, macro-F1 = **0.7318**.

---

## Quick start

### Install

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### Run the API

```bash
uvicorn src.api:app --port 8000
```

### Run Streamlit

```bash
streamlit run streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

By default the app connects to the API at `http://localhost:8000`. Override via environment variable:

```bash
API_URL=http://your-api-host:8000 streamlit run streamlit_app.py
```

---

## API

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model_info` | Model metadata |
| POST | `/predict` | Classify a single text |
| POST | `/predict_batch` | Classify up to 100 texts |

Example:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Отличный сервис, всё понравилось!"}'
```

```json
{"label": 1, "label_name": "positive", "probabilities": {"neutral": 0.003, "positive": 0.995, "negative": 0.001}}
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Project structure

```
streamlit_app.py       # Streamlit UI
src/
  api.py               # FastAPI service
  preprocessing.py     # text cleaning and lemmatisation
models/
  best_model.joblib    # trained model (sklearn Pipeline)
  model_metadata.json  # metrics and config
tests/                 # unit and integration tests
requirements.txt
```

---

## License

MIT
