from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("fake_news_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/user/detection', methods=['POST'])
def detect():
    data = request.get_json()
    news_text = data.get('text', '')
    if not news_text:
        return jsonify({"error": "No text provided"}), 400
    
    prediction = model.predict([news_text])[0]  # Fix method name
    probability = max(model.predict_proba([news_text])[0])  # Get probability score
    
    return jsonify({
        "prediction": "FAKE" if prediction == 1 else "TRUE",
        "probability": round(float(probability), 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if no PORT env variable is set
    app.run(host="0.0.0.0", port=port, debug=False)