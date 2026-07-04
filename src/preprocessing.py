import re
import os
import pandas as pd
from typing import Optional

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SAMPLE_SIZE, RANDOM_STATE

try:
	import spacy
	_HAS_SPACY = True
except Exception:
	spacy = None
	_HAS_SPACY = False

import nltk
from nltk.corpus import stopwords


def _ensure_nltk():
	try:
		stopwords.words("english")
	except LookupError:
		nltk.download("stopwords")


def load_reviews(path: Optional[str] = None) -> pd.DataFrame:
	path = path or RAW_DATA_PATH
	if not os.path.exists(path):
		raise FileNotFoundError(f"Raw data not found at {path}")
	df = pd.read_csv(path)
	return df


def clean_text(text: str, nlp=None, stop_words=None) -> str:
	if not isinstance(text, str):
		return ""
	text = text.lower()
	# remove urls
	text = re.sub(r"http\S+|www\.\S+", " ", text)
	# remove html
	text = re.sub(r"<.*?>", " ", text)
	# remove non-letters
	text = re.sub(r"[^a-z\s]", " ", text)
	# collapse whitespace
	text = re.sub(r"\s+", " ", text).strip()

	if stop_words is None:
		_ensure_nltk()
		stop_words = set(stopwords.words("english"))

	tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]

	if nlp is not None:
		# use spaCy lemmatization
		doc = nlp(" ".join(tokens))
		lemmas = [tok.lemma_ for tok in doc if tok.lemma_ not in stop_words and len(tok.lemma_) > 2]
		return " ".join(lemmas)

	return " ".join(tokens)


def preprocess_reviews(df: pd.DataFrame, sample_size: Optional[int] = SAMPLE_SIZE) -> pd.DataFrame:
	required_cols = ["content", "app_id", "review_id", "is_positive"]
	# ensure columns exist
	for c in ["content"]:
		if c not in df.columns:
			raise KeyError(f"Required column '{c}' not found in dataframe")

	# drop rows without content
	df = df.dropna(subset=["content"]).copy()

	# keep only necessary columns if present
	cols = [c for c in required_cols if c in df.columns]
	df = df[cols].copy()

	# sampling
	if sample_size is not None and sample_size > 0 and len(df) > sample_size:
		df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

	# load spaCy model if available
	nlp = None
	if _HAS_SPACY:
		try:
			nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
		except OSError:
			nlp = None

	_ensure_nltk()
	stop_words = set(stopwords.words("english"))

	df["clean_text"] = df["content"].astype(str).apply(lambda t: clean_text(t, nlp=nlp, stop_words=stop_words))

	return df


def save_cleaned_reviews(df: pd.DataFrame, path: Optional[str] = None) -> str:
	path = path or PROCESSED_DATA_PATH
	os.makedirs(os.path.dirname(path), exist_ok=True)
	df.to_csv(path, index=False)
	return path

