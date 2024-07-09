import os
import joblib
import sys
import json

# Get the absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and vectorizer from disk
model_path = os.path.join(script_dir, 'logistic_regression_model.pkl')
vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.pkl')

loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

def predict_news(model, vectorizer, news_text):
    X_new = vectorizer.transform([news_text])
    prediction = model.predict(X_new)
    probability = model.predict_proba(X_new)[:, 1]
    return prediction, probability

if __name__ == "__main__":
    news_text = sys.argv[1]
    prediction, probability = predict_news(loaded_model, loaded_vectorizer, news_text)
    result = {
        'prediction': 'FAKE' if prediction[0] == 1 else 'TRUE',
        'probability': probability[0]
    }
    print(json.dumps(result))
