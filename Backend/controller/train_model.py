import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from scipy.sparse import vstack

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Get the absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the datasets using absolute paths
fake_news_path = os.path.join(script_dir, 'Fake.csv')
true_news_path = os.path.join(script_dir, 'True.csv')

fake_news_df = pd.read_csv(fake_news_path)
true_news_df = pd.read_csv(true_news_path)

# Define preprocessing function
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

# Apply preprocessing to the text data
fake_news_df['clean_text'] = fake_news_df['title'].astype(str) + ' ' + fake_news_df['text'].astype(str)
fake_news_df['clean_text'] = fake_news_df['clean_text'].apply(preprocess_text)

true_news_df['clean_text'] = true_news_df['title'].astype(str) + ' ' + true_news_df['text'].astype(str)
true_news_df['clean_text'] = true_news_df['clean_text'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer and transform the data
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(pd.concat([fake_news_df['clean_text'], true_news_df['clean_text']], axis=0))
X_fake = X[:len(fake_news_df)]
X_true = X[len(fake_news_df):]
X_train = vstack([X_fake, X_true])
y_train = pd.concat([pd.Series([1] * len(fake_news_df)), pd.Series([0] * len(true_news_df))], axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model and vectorizer to disk
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
