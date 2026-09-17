import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from .config import FIGURES_DIR


def _ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_top_words(top_words_df: pd.DataFrame, filename: str = "top_words.png") -> str:
    """Plot the most important words from the vocabulary."""
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    df = top_words_df.copy().sort_values("score", ascending=True)
    ax.barh(df["word"], df["score"], color="C0")
    ax.set_xlabel("Score")
    ax.set_title("Top Words")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out)
    plt.close(fig)
    return out


def create_wordcloud(texts: List[str], filename: str = "wordcloud.png") -> str:
    """Create and save a word cloud from a collection of cleaned reviews."""
    _ensure_figures_dir()
    text = " ".join(texts) if isinstance(texts, list) else str(texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sentiment_distribution(df: pd.DataFrame, filename: str = "sentiment_distribution.png") -> str:
    """Plot the distribution of positive and negative reviews."""
    _ensure_figures_dir()
    if "is_positive" not in df.columns:
        raise KeyError("DataFrame must contain 'is_positive' column for sentiment distribution plot")

    counts = df["is_positive"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["C2", "C3"])
    ax.set_xticklabels([str(value) for value in counts.index])
    ax.set_xlabel("is_positive")
    ax.set_ylabel("count")
    ax.set_title("Sentiment Distribution")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_topic_frequency(
    topic_distribution_df: pd.DataFrame,
    filename: str = "topic_frequency.png",
    use_percentage: bool = False,
) -> str:
    """Plot dominant topic frequencies or their percentage shares."""
    _ensure_figures_dir()
    value_column = "topic_share_percent" if use_percentage else "topic_frequency"
    required_columns = {"topic_number", value_column}
    if required_columns - set(topic_distribution_df.columns):
        raise KeyError(f"DataFrame must contain 'topic_number' and '{value_column}' columns")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(topic_distribution_df["topic_number"].astype(str), topic_distribution_df[value_column], color="C1")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Share (%)" if use_percentage else "Frequency")
    ax.set_title("Topic Share" if use_percentage else "Topic Frequency")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out)
    plt.close(fig)
    return out

