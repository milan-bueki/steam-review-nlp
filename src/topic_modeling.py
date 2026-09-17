from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF

from .config import N_TOPICS


def run_lda(bow_matrix, n_topics: int = N_TOPICS, random_state: int = 42) -> LatentDirichletAllocation:
    """Fit an LDA model to the bag-of-words matrix."""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda.fit(bow_matrix)
    return lda


def run_nmf(tfidf_matrix, n_topics: int = N_TOPICS, random_state: int = 42) -> NMF:
    """Fit an NMF model to the TF-IDF matrix."""
    nmf = NMF(n_components=n_topics, random_state=random_state)
    nmf.fit(tfidf_matrix)
    return nmf


def extract_topics(model: Any, vectorizer, n_words: int = 15, model_name: str = "model") -> pd.DataFrame:
    """Extract the top words for each topic and return them as a DataFrame."""
    if not hasattr(model, "components_"):
        raise ValueError("Model does not have components_")

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        ranked = sorted(zip(feature_names, topic), key=lambda item: item[1], reverse=True)[:n_words]
        top_words = ", ".join(word for word, _ in ranked)
        topics.append({"model": model_name, "topic_number": int(topic_idx), "top_words": top_words})

    return pd.DataFrame(topics)


def compute_topic_distribution(model: Any, matrix, n_topics: int = N_TOPICS) -> pd.DataFrame:
    """Assign each document to its dominant topic and summarize the distribution."""
    topic_probabilities = model.transform(matrix)
    dominant_topic = topic_probabilities.argmax(axis=1)
    counts = pd.Series(dominant_topic).value_counts().reindex(range(n_topics), fill_value=0)
    total_reviews = len(dominant_topic)
    topic_shares = (counts / total_reviews * 100).round(2) if total_reviews else counts.astype(float)

    return pd.DataFrame(
        {
            "topic_number": counts.index,
            "topic_frequency": counts.values,
            "topic_share_percent": topic_shares.values,
        }
    )
