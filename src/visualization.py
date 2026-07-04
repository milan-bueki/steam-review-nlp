import os
from typing import List
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

from .config import FIGURES_DIR


def _ensure_figures_dir():
	os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_top_words(top_words_df: pd.DataFrame, filename: str = "top_words.png") -> str:
	_ensure_figures_dir()
	fig, ax = plt.subplots(figsize=(10, 6))
	df = top_words_df.copy()
	df = df.sort_values("score", ascending=True)
	ax.barh(df["word"], df["score"], color="C0")
	ax.set_xlabel("Score")
	ax.set_title("Top Words")
	plt.tight_layout()
	out = os.path.join(FIGURES_DIR, filename)
	fig.savefig(out)
	plt.close(fig)
	return out


def create_wordcloud(texts: List[str], filename: str = "wordcloud.png") -> str:
	_ensure_figures_dir()
	if isinstance(texts, list):
		text = " ".join(texts)
	else:
		text = str(texts)
	wc = WordCloud(width=800, height=400, background_color="white").generate(text)
	fig = plt.figure(figsize=(12, 6))
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	out = os.path.join(FIGURES_DIR, filename)
	fig.savefig(out, bbox_inches="tight")
	plt.close(fig)
	return out


def plot_sentiment_distribution(df: pd.DataFrame, filename: str = "sentiment_distribution.png") -> str:
	_ensure_figures_dir()
	if "is_positive" not in df.columns:
		raise KeyError("DataFrame must contain 'is_positive' column for sentiment distribution plot")
	counts = df["is_positive"].value_counts().sort_index()
	fig, ax = plt.subplots(figsize=(6, 4))
	counts.plot(kind="bar", ax=ax, color=["C2", "C3"])
	ax.set_xticklabels([str(x) for x in counts.index])
	ax.set_xlabel("is_positive")
	ax.set_ylabel("count")
	ax.set_title("Sentiment Distribution")
	plt.tight_layout()
	out = os.path.join(FIGURES_DIR, filename)
	fig.savefig(out)
	plt.close(fig)
	return out

