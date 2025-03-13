from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load the sentence transformer once (smaller model)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "ensemble_model.pkl")
vectorizer_path = os.path.join(script_dir, "tfidf_vectorizer.pkl")

def get_bert_embedding(text):
    return sentence_model.encode([text], convert_to_numpy=True).flatten()

def predict_news(news_text):
    try:
        # Load the model and vectorizer only when needed
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)

        tfidf_features = loaded_vectorizer.transform([news_text]).toarray()
        bert_features = get_bert_embedding(news_text).reshape(1, -1)
        combined_features = np.hstack((tfidf_features, bert_features))

        prediction = loaded_model.predict(combined_features)
        probability = loaded_model.predict_proba(combined_features)[:, 1]
        
        return prediction[0], probability[0]
    except Exception as e:
        return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/user/detection', methods=['POST'])
def detect():
    data = request.get_json()
    news_text = data.get('text', '')
    if not news_text:
        return jsonify({"error": "No text provided"}), 400
    
    prediction, probability = predict_news(news_text)
    if prediction is None:
        return jsonify({"error": "Prediction failed"}), 500
    
    return jsonify({
        "prediction": "FAKE" if prediction == 1 else "TRUE",
        "probability": round(float(probability), 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if no PORT env variable is set
    app.run(host="0.0.0.0", port=port, debug=False)
