# Sentiment Analyzer

**English** | [Русский](README.ru.md)

Web application for sentiment analysis of short Russian-language texts (up to 5000 characters). Classifies messages into three categories: **positive / neutral / negative**.

Model: TF-IDF + bigrams + logistic regression, macro-F1 = **0.7318**.

---

## Quick start

### Install

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### Run

```bash
streamlit run streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

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
  preprocessing.py     # text cleaning and lemmatisation
models/
  best_model.joblib    # trained model (sklearn Pipeline)
  model_metadata.json  # metrics and config
tests/                 # unit tests
requirements.txt
```

---

## License

MIT
