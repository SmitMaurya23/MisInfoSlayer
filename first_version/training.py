import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import gdown

# Google Drive file IDs
fake_news_file_id = '1r2WUuMsiEbcn_j95A-ylvH8HC9yItaUk'
true_news_file_id = '1EqIUB36qMs19CaejQsyPQ_qUw9qlX6V5'

# Download datasets
gdown.download(f'https://drive.google.com/uc?id={fake_news_file_id}', 'Fake.csv', quiet=False)
gdown.download(f'https://drive.google.com/uc?id={true_news_file_id}', 'True.csv', quiet=False)

# Load datasets efficiently
use_cols = ['title', 'text']  # Load only necessary columns
fake_df = pd.read_csv('Fake.csv', usecols=use_cols, dtype=str, low_memory=False).fillna('')
true_df = pd.read_csv('True.csv', usecols=use_cols, dtype=str, low_memory=False).fillna('')
# Labeling data
fake_df["label"] = 1  # Fake News
true_df["label"] = 0  # True News

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning
df["text"] = df["text"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Build model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model
joblib.dump(pipeline, "fake_news_model.pkl")
print("Model saved as fake_news_model.pkl")
