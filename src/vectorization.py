from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import MAX_FEATURES, NGRAM_RANGE, VECT_MAX_DF, VECT_MIN_DF


def create_bow_matrix(
    texts: List[str],
    max_features: int = MAX_FEATURES,
    min_df: int = VECT_MIN_DF,
    max_df: float = VECT_MAX_DF,
):
    """Create a bag-of-words matrix with unigrams and bigrams."""
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=NGRAM_RANGE)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def create_tfidf_matrix(
    texts: List[str],
    max_features: int = MAX_FEATURES,
    min_df: int = VECT_MIN_DF,
    max_df: float = VECT_MAX_DF,
):
    """Create a TF-IDF matrix with unigrams and bigrams."""
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=NGRAM_RANGE)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def get_top_words(vectorizer, matrix, n: int = 20) -> pd.DataFrame:
    """Return the top n words or ngrams by total weight in the matrix."""
    if hasattr(matrix, "toarray"):
        arr = matrix.toarray()
    else:
        arr = matrix

    scores = arr.sum(axis=0)
    if hasattr(scores, "A1"):
        scores = scores.A1

    feature_names = vectorizer.get_feature_names_out()
    indices = scores.argsort()[::-1][:n]
    top = [(feature_names[index], float(scores[index])) for index in indices]
    return pd.DataFrame(top, columns=["word", "score"])

