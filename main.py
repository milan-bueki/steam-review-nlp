import pandas as pd

from src import preprocessing, topic_modeling, vectorization, visualization
from src.config import N_TOP_WORDS, PROCESSED_DATA_PATH
from src.utils import create_directories, save_table


def main() -> None:
    print("Creating directories...")
    create_directories()

    try:
        print("Loading raw reviews...")
        df = preprocessing.load_reviews()
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return

    try:
        print("Preprocessing reviews (this may take a while)...")
        cleaned = preprocessing.preprocess_reviews(df)
        print(f"Saving cleaned reviews to {PROCESSED_DATA_PATH}...")
        preprocessing.save_cleaned_reviews(cleaned)
    except Exception as error:
        print(f"Preprocessing error: {error}")
        return

    texts = cleaned["clean_text"].fillna("").astype(str).tolist()

    print("Creating Bag-of-Words matrix...")
    bow_vectorizer, bow_matrix = vectorization.create_bow_matrix(texts)
    print("Creating TF-IDF matrix...")
    tfidf_vectorizer, tfidf_matrix = vectorization.create_tfidf_matrix(texts)

    print("Computing top words (BoW)...")
    top_words = vectorization.get_top_words(bow_vectorizer, bow_matrix, n=N_TOP_WORDS)
    save_table(top_words, "top_words_bow.csv")

    print("Running LDA...")
    lda = topic_modeling.run_lda(bow_matrix)
    lda_topics = topic_modeling.extract_topics(lda, bow_vectorizer, n_words=15, model_name="LDA")
    save_table(lda_topics, "topics_lda.csv")

    print("Running NMF...")
    nmf = topic_modeling.run_nmf(tfidf_matrix)
    nmf_topics = topic_modeling.extract_topics(nmf, tfidf_vectorizer, n_words=15, model_name="NMF")
    save_table(nmf_topics, "topics_nmf.csv")

    print("Saving compact topic keyword tables...")
    lda_keywords = topic_modeling.extract_topics(lda, bow_vectorizer, n_words=10, model_name="LDA")
    topic_keywords = lda_keywords[["topic_number", "top_words"]].copy()
    topic_keywords = topic_keywords.rename(columns={"top_words": "keywords"})
    save_table(topic_keywords, "topic_keywords.csv")

    print("Computing topic distribution...")
    topic_distribution = topic_modeling.compute_topic_distribution(lda, bow_matrix)
    save_table(topic_distribution, "topic_distribution.csv")

    print("Generating visualizations...")
    visualization.plot_top_words(top_words)
    visualization.create_wordcloud(texts[:5000])
    visualization.plot_sentiment_distribution(cleaned)
    visualization.plot_topic_frequency(topic_distribution)

    print("Pipeline finished.")


if __name__ == "__main__":
    main()

