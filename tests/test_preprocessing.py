"""Unit tests for the text preprocessor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pytest

from src.preprocessing import TextPreprocessor


@pytest.fixture(scope="module")
def preproc() -> TextPreprocessor:
    return TextPreprocessor()


def _process(p: TextPreprocessor, text: str) -> str:
    return p.transform([text])[0]


def test_url_removed(preproc):
    out = _process(preproc, "купил тут http://example.com круто")
    assert "http" not in out
    assert "example" not in out


def test_hashtag_removed_with_word(preproc):
    out = _process(preproc, "супер #отзыв класс")
    assert "отзыв" not in out


def test_mention_removed(preproc):
    out = _process(preproc, "@user привет")
    assert "user" not in out


def test_emoji_preserved(preproc):
    out = _process(preproc, "норм 😀")
    assert "😀" in out


def test_negation_preserved(preproc):
    out = _process(preproc, "это не плохо")
    tokens = out.split()
    assert "не" in tokens


def test_stopword_removed(preproc):
    out = _process(preproc, "это и так понятно")
    tokens = out.split()
    assert "и" not in tokens


def test_repeated_chars_collapsed(preproc):
    out = _process(preproc, "приветттт")
    assert "привет" in out.split()


def test_lemmatization(preproc):
    out = _process(preproc, "купил красивое пальто")
    tokens = out.split()
    assert "купить" in tokens
    assert "красивый" in tokens
    assert "пальто" in tokens


def test_empty_string(preproc):
    out = _process(preproc, "")
    assert out == ""


def test_only_noise(preproc):
    out = _process(preproc, "@user http://x.co #тег")
    assert out == ""


def test_picklable_roundtrip(preproc):
    sample = "Привееееет!!! @vasya http://t.co/x #круто это не плохо 😀"
    expected = preproc.transform([sample])[0]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "preproc.joblib"
        joblib.dump(preproc, path)
        loaded = joblib.load(path)

    actual = loaded.transform([sample])[0]
    assert actual == expected
