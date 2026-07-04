from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import MAX_FEATURES, VECT_MIN_DF, VECT_MAX_DF


def create_bow_matrix(texts: List[str], max_features: int = MAX_FEATURES, min_df: int = VECT_MIN_DF, max_df: float = VECT_MAX_DF):
	vect = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
	X = vect.fit_transform(texts)
	return vect, X


def create_tfidf_matrix(texts: List[str], max_features: int = MAX_FEATURES, min_df: int = VECT_MIN_DF, max_df: float = VECT_MAX_DF):
	vect = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
	X = vect.fit_transform(texts)
	return vect, X


def get_top_words(vectorizer, matrix, n: int = 20) -> pd.DataFrame:
	"""Return top n words by global frequency (for BoW) or tf-idf sum.

	Returns a DataFrame with columns `word` and `score`.
	"""
	import numpy as np

	if hasattr(matrix, "toarray"):
		arr = matrix.toarray()
	else:
		arr = matrix

	scores = arr.sum(axis=0)
	if hasattr(scores, "A1"):
		scores = scores.A1
	feature_names = vectorizer.get_feature_names_out()
	idx = (-scores).argsort()[:n]
	top = [(feature_names[i], float(scores[i])) for i in idx]
	return pd.DataFrame(top, columns=["word", "score"])

