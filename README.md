# MisInfoSlayer

This repository contains a **Fake News Detection Model** that uses **Machine Learning and NLP** techniques to classify news articles as "FAKE" or "TRUE". The model combines **TF-IDF** and **BERT embeddings** to create a robust feature representation and is trained using a **Random Forest Classifier**. The Flask-based web API allows users to input text and receive a prediction in real time.

## Hosted Website
The Fake News Detection model is hosted at: [(https://misinfoslayer.onrender.com/)](#) 

## Features
- **Preprocessing:** Cleans and processes news articles.
- **TF-IDF Vectorization:** Extracts important words and their importance in text classification.
- **BERT Embeddings:** Captures the contextual meaning of news articles.
- **Machine Learning Model:** Uses Random Forest for classification.
- **Flask API:** Provides an endpoint for real-time fake news detection.

---
## 1. Running the Setup Locally

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.x
- pip
- Git
- CUDA-enabled GPU (optional but recommended for faster BERT processing)

### Clone the Repository
```bash
git clone https://github.com/SmitMaurya23/MisInfoSlayer.git
cd MisInfoSlayer
```

### Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Download Datasets
The dataset is automatically downloaded via `gdown` when you run the `training.py` script.
However, if needed, you can manually download the files from Google Drive:
- Fake News: [Google Drive Link](https://drive.google.com/file/d/1r2WUuMsiEbcn_j95A-ylvH8HC9yItaUk/view)
- True News: [Google Drive Link](https://drive.google.com/file/d/1EqIUB36qMs19CaejQsyPQ_qUw9qlX6V5/view)

Save them as `Fake.csv` and `True.csv` in the project directory.

### 2. Training the Model
Run the training script to preprocess data, train the classifier, and save the model components.
```bash
python training.py
```
This will:
- Preprocess text using NLTK and BeautifulSoup.
- Generate **TF-IDF features**.
- Generate **BERT embeddings** using `sentence-transformers`.
- Train a **Random Forest Classifier**.
- Save the model and its components for later use.

### 3. Running the Flask API
Once the model is trained, start the Flask server:
```bash
python app.py
```
The API will be available at: `http://0.0.0.0:10000/`

### 4. Testing the API
To test the API locally, send a POST request with JSON data:
```bash
curl -X POST http://127.0.0.1:10000/api/user/detection \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a sample news article."}'
```
Expected response:
```json
{
    "prediction": "FAKE",
    "probability": 0.87
}
```


## Technologies Used
- **Python, Flask** – Backend API
- **Pandas, NumPy, Scikit-learn** – Data processing & model training
- **NLTK, BeautifulSoup** – Text preprocessing
- **TF-IDF, BERT** – Feature extraction
- **Random Forest Classifier** – ML model for classification
- **SentenceTransformers** – Efficient embedding generation



