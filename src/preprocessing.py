import os
import re
from typing import Optional, Set

import nltk
import pandas as pd
from nltk.corpus import stopwords

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH, RANDOM_STATE, SAMPLE_SIZE

try:
    import spacy

    _HAS_SPACY = True
except Exception:
    spacy = None
    _HAS_SPACY = False

CUSTOM_STOPWORDS = {
    "game",
    "games",
    "play",
    "played",
    "player",
    "players",
    "one",
    "get",
    "got",
    "really",
    "good",
    "bad",
    "great",
    "nice",
    "yes",
    "still",
    "even",
    "much",
    "time",
    "like",
    "would",
    "also",
    "can",
    "dont",
    "didnt",
    "ive",
    "im",
    "steam",
    "valve",
    "shit",
    "fuck",
    "fucking",
    "damn",
    "lol",
    "lmao",
    "wtf",
    "yeah",
    "nope",
    "okay",
    "ok",
    "awesome",
    "amazing",
    "best",
    "worst",
    "love",
    "hate",
    "half",
    "life",
    "portal",
    "counter",
    "strike",
    "csgo",
    "source",
    "engine",
}


def _ensure_nltk() -> None:
    """Download stopwords if they are not present locally."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def _get_stop_words(stop_words: Optional[Set[str]] = None) -> Set[str]:
    """Return the combined set of default and custom stopwords."""
    _ensure_nltk()
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    else:
        stop_words = set(stop_words)
    return stop_words.union(CUSTOM_STOPWORDS)


def load_reviews(path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw review dataset from disk."""
    path = path or RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_csv(path)


def clean_text(text: str, nlp=None, stop_words: Optional[Set[str]] = None) -> str:
    """Clean a single review and return lemmatized tokens as a string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    phrase_patterns = [
        r"\bhalf\s+life\b",
        r"\bcounter\s+strike\b",
        r"\bcsgo\b",
        r"\bsource\s+engine\b",
        r"\bsteam\b",
        r"\bvalve\b",
    ]
    for pattern in phrase_patterns:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_words = _get_stop_words(stop_words)
    tokens = [token for token in text.split() if token not in stop_words and len(token) > 2]

    if nlp is not None:
        doc = nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc if token.lemma_.lower() not in stop_words and len(token.lemma_) > 2]
        return " ".join(lemmas)

    return " ".join(tokens)


def preprocess_reviews(df: pd.DataFrame, sample_size: Optional[int] = SAMPLE_SIZE) -> pd.DataFrame:
    """Clean the review dataset and return a DataFrame with a `clean_text` column."""
    if "content" not in df.columns:
        raise KeyError("Required column 'content' not found in dataframe")

    df = df.dropna(subset=["content"]).copy()

    available_columns = [column for column in ["app_id", "review_id", "is_positive", "content"] if column in df.columns]
    df = df[available_columns].copy()

    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    nlp = None
    if _HAS_SPACY:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            nlp = None

    stop_words = _get_stop_words()
    df["clean_text"] = df["content"].astype(str).apply(lambda text: clean_text(text, nlp=nlp, stop_words=stop_words))
    df["clean_text"] = df["clean_text"].fillna("")
    return df


def save_cleaned_reviews(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """Save the cleaned reviews to disk."""
    path = path or PROCESSED_DATA_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

