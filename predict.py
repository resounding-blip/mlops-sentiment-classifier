import joblib
import argparse

def predict_sentiment(text: str, model_path: str):
    """Predicts the sentiment of a given text using a saved model."""
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return

    prediction = model.predict([text])
    sentiment = "positive" if prediction[0] == 1 else "negative"

    print(f"\n--- Prediction Result ---")
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {sentiment}")
    print("-------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the sentiment of a given text.")
    parser.add_argument("text", type=str, help="The text to analyze.")
    parser.add_argument("--model_path", type=str, default="sentiment_model.pkl", help="Path to the trained model.")
    args = parser.parse_args()

    predict_sentiment(args.text, args.model_path)