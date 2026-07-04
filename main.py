from src.utils import create_directories, save_table
from src import preprocessing, vectorization, topic_modeling, visualization
from src.config import PROCESSED_DATA_PATH, N_TOP_WORDS
import pandas as pd


def main():
	print("Creating directories...")
	create_directories()

	try:
		print("Loading raw reviews...")
		df = preprocessing.load_reviews()
	except FileNotFoundError as e:
		print(f"Error: {e}")
		return

	try:
		print("Preprocessing reviews (this may take a while)...")
		cleaned = preprocessing.preprocess_reviews(df)
		print(f"Saving cleaned reviews to {PROCESSED_DATA_PATH}...")
		preprocessing.save_cleaned_reviews(cleaned)
	except Exception as e:
		print(f"Preprocessing error: {e}")
		return

	texts = cleaned["clean_text"].fillna("").tolist()

	print("Creating Bag-of-Words matrix...")
	bow_vect, bow = vectorization.create_bow_matrix(texts)
	print("Creating TF-IDF matrix...")
	tfidf_vect, tfidf = vectorization.create_tfidf_matrix(texts)

	print("Computing top words (BoW)...")
	top_words = vectorization.get_top_words(bow_vect, bow, n=N_TOP_WORDS)
	table_path = save_table(top_words, "top_words_bow.csv")
	print(f"Top words saved to {table_path}")

	print("Running LDA...")
	lda = topic_modeling.run_lda(bow)
	lda_topics = topic_modeling.extract_topics(lda, bow_vect, n_words=10, model_name="LDA")
	lda_path = save_table(lda_topics, "topics_lda.csv")
	print(f"LDA topics saved to {lda_path}")

	print("Running NMF...")
	nmf = topic_modeling.run_nmf(tfidf)
	nmf_topics = topic_modeling.extract_topics(nmf, tfidf_vect, n_words=10, model_name="NMF")
	nmf_path = save_table(nmf_topics, "topics_nmf.csv")
	print(f"NMF topics saved to {nmf_path}")

	print("Generating visualizations...")
	top_words_fig = visualization.plot_top_words(top_words)
	wc_fig = visualization.create_wordcloud(texts[:5000])
	senti_fig = visualization.plot_sentiment_distribution(cleaned)
	print(f"Saved figures: {top_words_fig}, {wc_fig}, {senti_fig}")

	print("Pipeline finished.")


if __name__ == "__main__":
	main()

