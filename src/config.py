import os

# Paths
RAW_DATA_PATH = os.path.join("data", "raw", "output.csv")
STEAMSPY_PATH = os.path.join("data", "raw", "output_steamspy.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_reviews.csv")
FIGURES_DIR = os.path.join("reports", "figures")
TABLES_DIR = os.path.join("reports", "tables")
MODELS_DIR = os.path.join("models")

# Parameters
SAMPLE_SIZE = 50000
RANDOM_STATE = 42
N_TOP_WORDS = 20
N_TOPICS = 5
MAX_FEATURES = 5000

# Vectorizer defaults
VECT_MIN_DF = 5
VECT_MAX_DF = 0.8
