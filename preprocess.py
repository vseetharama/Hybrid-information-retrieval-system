import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


REQUIRED_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet.zip": "wordnet",
    "corpora/omw-1.4.zip": "omw-1.4",
}


def ensure_nltk_resources() -> None:
    for resource_path, package in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                nltk.data.find(resource_path)
            except LookupError as error:
                raise RuntimeError(
                    f"NLTK resource '{package}' is unavailable. "
                    "Install it with nltk.download()."
                ) from error


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    ensure_nltk_resources()
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_resources()
    return WordNetLemmatizer()


def preprocess(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    ensure_nltk_resources()
    normalized = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(normalized)
    return [
        _lemmatizer().lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in _stop_words()
    ]


def preprocess_to_text(text: str) -> str:
    return " ".join(preprocess(text))
