import pandas as pd

# Load the datasets (assuming you've already loaded them as shown previously)
fake_news_df = pd.read_csv('Fake.csv')
true_news_df = pd.read_csv('True.csv')

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



