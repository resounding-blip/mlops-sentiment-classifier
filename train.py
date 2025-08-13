import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import argparse

def train_model(data_path, model_path):
    """Trains a sentiment analysis model from a CSV file."""
    print("--- 🚀 Starting model training ---")

    # Load data
    df = pd.read_csv(data_path)

    # Map labels to binary
    label_map = {'positive': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(label_map)

    X = df['text']
    y = df['label']

    # Create a model pipeline
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SGDClassifier(random_state=42)),
    ])

    print("Training the model...")
    model_pipeline.fit(X, y)

    # Save the trained model
    joblib.dump(model_pipeline, model_path)
    print(f"--- ✅ Model saved to {model_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument("--data_path", type=str, default="reviews.csv", help="Path to the training data CSV.")
    parser.add_argument("--model_path", type=str, default="sentiment_model.pkl", help="Path to save the trained model.")
    args = parser.parse_args()

    train_model(args.data_path, args.model_path)