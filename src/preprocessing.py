from __future__ import annotations

import re
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)
_MENTION_RE = re.compile(r"@\w+", flags=re.UNICODE)
_DEDUPE_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

# Letter token: Russian (incl. ё) or Latin. Emoji — separate class covering
# misc symbols/dingbats, emoticons, supplemental symbols, pictographs A/B,
# and transport/map Unicode ranges.
_TOKEN_RE = re.compile(
    r"[а-яёa-z]+"
    r"|[☀-➿"
    r"\U0001f300-\U0001f5ff"
    r"\U0001f600-\U0001f64f"
    r"\U0001f680-\U0001f6ff"
    r"\U0001f900-\U0001f9ff"
    r"\U0001fa00-\U0001faff]",
    flags=re.UNICODE,
)

NEGATIONS = frozenset({"не", "нет", "ни", "без", "нельзя"})


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        remove_urls: bool = True,
        remove_hashtags: bool = True,
        remove_mentions: bool = True,
        lowercase: bool = True,
        dedupe_chars: bool = True,
        lemmatize: bool = True,
        remove_stopwords: bool = True,
        keep_negations: bool = True,
        keep_emoji: bool = True,
    ) -> None:
        self.remove_urls = remove_urls
        self.remove_hashtags = remove_hashtags
        self.remove_mentions = remove_mentions
        self.lowercase = lowercase
        self.dedupe_chars = dedupe_chars
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.keep_emoji = keep_emoji
        self._morph = None
        self._stop: set[str] | None = None

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X: Iterable[str]) -> list[str]:  # noqa: N803
        self._ensure_resources()
        return [self._process_one(text) for text in X]

    # Heavy resources are not serialised; they are initialised on first transform call
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_morph"] = None
        state["_stop"] = None
        return state

    def _ensure_resources(self) -> None:
        if self.lemmatize and self._morph is None:
            # pymorphy2 0.9.1 calls inspect.getargspec which was removed in Python 3.11.
            # This shim redirects to getfullargspec without changing behaviour or
            # OpenCorpora dictionaries.
            import inspect as _inspect

            if not hasattr(_inspect, "getargspec"):

                def _getargspec(func):  # type: ignore[no-redef]
                    spec = _inspect.getfullargspec(func)
                    return (spec.args, spec.varargs, spec.varkw, spec.defaults)

                _inspect.getargspec = _getargspec  # type: ignore[attr-defined]
            import pymorphy2

            self._morph = pymorphy2.MorphAnalyzer()
        if self.remove_stopwords and self._stop is None:
            from nltk.corpus import stopwords

            stop = set(stopwords.words("russian"))
            if self.keep_negations:
                stop -= NEGATIONS
            self._stop = stop

    def _clean(self, text: str) -> str:
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        if self.remove_hashtags:
            text = _HASHTAG_RE.sub(" ", text)
        if self.remove_mentions:
            text = _MENTION_RE.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        if self.dedupe_chars:
            text = _DEDUPE_RE.sub(r"\1", text)
        return text

    def _process_one(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        text = self._clean(text)
        tokens = _TOKEN_RE.findall(text)

        out: list[str] = []
        for tok in tokens:
            is_letters = tok[0].isalpha()
            if is_letters:
                if self.lemmatize:
                    tok = self._morph.parse(tok)[0].normal_form
                if self.remove_stopwords and tok in self._stop:
                    continue
                out.append(tok)
            else:
                if self.keep_emoji:
                    out.append(tok)

        return " ".join(out)
