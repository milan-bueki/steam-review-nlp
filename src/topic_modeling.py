import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from typing import Tuple

from .config import N_TOPICS


def run_lda(bow_matrix, n_topics: int = N_TOPICS, random_state: int = 42) -> LatentDirichletAllocation:
	lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
	lda.fit(bow_matrix)
	return lda


def run_nmf(tfidf_matrix, n_topics: int = N_TOPICS, random_state: int = 42) -> NMF:
	nmf = NMF(n_components=n_topics, random_state=random_state)
	nmf.fit(tfidf_matrix)
	return nmf


def extract_topics(model, vectorizer, n_words: int = 10, model_name: str = "model") -> pd.DataFrame:
	feature_names = vectorizer.get_feature_names_out()
	topics = []
	if hasattr(model, "components_"):
		comps = model.components_
	else:
		raise ValueError("Model does not have components_")

	for topic_idx, topic in enumerate(comps):
		top_idx = topic.argsort()[-n_words:][::-1]
		top_words = [feature_names[i] for i in top_idx]
		topics.append({"model": model_name, "topic_number": int(topic_idx), "top_words": ", ".join(top_words)})

	return pd.DataFrame(topics)
