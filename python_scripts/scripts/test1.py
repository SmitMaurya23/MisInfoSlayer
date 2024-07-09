# %%
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# %%
# Load the datasets
fake_news_df = pd.read_csv('Fake.csv')
true_news_df = pd.read_csv('True.csv')


# %%
# Display basic information about the datasets
print("Fake News Dataset Info:")
print(fake_news_df.info())

print("\nTrue News Dataset Info:")
print(true_news_df.info())

# Display the first few rows of each dataset
print("\nFirst few rows of Fake News Dataset:")
print(fake_news_df.head())

print("\nFirst few rows of True News Dataset:")
print(true_news_df.head())


# %%
# Define preprocessing function
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercase and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Apply preprocessing to Title and Text columns in both datasets
fake_news_df['clean_text'] = fake_news_df['title'].astype(str) + ' ' + fake_news_df['text'].astype(str)
fake_news_df['clean_text'] = fake_news_df['clean_text'].apply(preprocess_text)

true_news_df['clean_text'] = true_news_df['title'].astype(str) + ' ' + true_news_df['text'].astype(str)
true_news_df['clean_text'] = true_news_df['clean_text'].apply(preprocess_text)

# Display the first few rows of preprocessed data
print("Preprocessed Fake News Data:")
print(fake_news_df[['clean_text']].head())

print("\nPreprocessed True News Data:")
print(true_news_df[['clean_text']].head())


# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed

# Fit and transform the training data (Fake and True news combined)
X = vectorizer.fit_transform(pd.concat([fake_news_df['clean_text'], true_news_df['clean_text']], axis=0))

# Separate back into fake and true news
X_fake = X[:len(fake_news_df)]
X_true = X[len(fake_news_df):]

# Combine back the sparse matrices
X_train = vstack([X_fake, X_true])

# Create labels: 1 for fake news, 0 for true news
y_train = pd.concat([pd.Series([1] * len(fake_news_df)), pd.Series([0] * len(true_news_df))], axis=0)

# Optionally, you can also split the data into training and testing sets
# Example: Splitting into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Check the shape of the final feature matrix and labels
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)



# %%
print(X_train)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display more detailed evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model on the training data
rf_model.fit(X_train, y_train)

# Predict on the testing data
rf_y_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Display classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))


# %%
import joblib

# Save the model to disk
model_filename = 'logistic_regression_model.pkl'
joblib.dump(model, model_filename)

# Save the TF-IDF vectorizer to disk
vectorizer_filename = 'tfidf_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_filename)


# %%
# Load the model from disk
loaded_model = joblib.load(model_filename)

# Load the TF-IDF vectorizer from disk
loaded_vectorizer = joblib.load(vectorizer_filename)


# %%
def predict_news(model, vectorizer, news_text):
    # Preprocess the news_text using the loaded vectorizer
    X_new = vectorizer.transform([news_text])
    
    # Predict using the loaded model
    prediction = model.predict(X_new)
    probability = model.predict_proba(X_new)[:, 1]  # Probability of being fake news (class 1)
    
    return prediction, probability


# %%
# Example usage
news_text = "The Joint Entrance Exam (JEE)-mains results were released Sunday morning, with Ved Lahoti from the IIT Delhi zone achieving the top rank by scoring 355 out of 360 marks. Out of the 48,248 candidates who qualified for admission to the IITs, 7,964 are female."

prediction, probability = predict_news(loaded_model, loaded_vectorizer, news_text)

if prediction[0] == 1:
    print(f"The news is predicted as FAKE with probability {probability[0]:.2f}.")
else:
    print(f"The news is predicted as TRUE with probability {1 - probability[0]:.2f}.")



