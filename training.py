import os
import gdown
import pandas as pd
import re
import torch
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib
from scipy.sparse import hstack
from transformers import BertTokenizer, BertModel
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Google Drive file IDs
fake_news_file_id = '1r2WUuMsiEbcn_j95A-ylvH8HC9yItaUk'
true_news_file_id = '1EqIUB36qMs19CaejQsyPQ_qUw9qlX6V5'

# Download datasets
gdown.download(f'https://drive.google.com/uc?id={fake_news_file_id}', 'Fake.csv', quiet=False)
gdown.download(f'https://drive.google.com/uc?id={true_news_file_id}', 'True.csv', quiet=False)

# Load datasets efficiently
use_cols = ['title', 'text']  # Load only necessary columns
fake_news_df = pd.read_csv('Fake.csv', usecols=use_cols, dtype=str, low_memory=False).fillna('')
true_news_df = pd.read_csv('True.csv', usecols=use_cols, dtype=str, low_memory=False).fillna('')

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+|@\S+|\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

nltk.download('punkt_tab')
fake_news_df['clean_text'] = (fake_news_df['title'] + ' ' + fake_news_df['text']).apply(preprocess_text)
true_news_df['clean_text'] = (true_news_df['title'] + ' ' + 
true_news_df['text']).apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_tfidf = vectorizer.fit_transform(pd.concat([fake_news_df['clean_text'], true_news_df['clean_text']], axis=0))

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device).half()
bert_model.half()
bert_model.eval()


@torch.no_grad()
def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = bert_tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
X_bert = sentence_model.encode(pd.concat([fake_news_df['clean_text'], true_news_df['clean_text']], axis=0).tolist(), batch_size=64)
np.save('X_bert.npy', X_bert)


# Combine Features
X = hstack([X_tfidf, X_bert])
y = np.concatenate([np.ones(len(fake_news_df)), np.zeros(len(true_news_df))])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save Model and Components
joblib.dump(model, 'ensemble_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
bert_tokenizer.save_pretrained('bert_tokenizer')
bert_model.save_pretrained('bert_model')
sentence_model.save('sentence_model')
