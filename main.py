import re
import nltk
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')  # WordNet lemmatizer requires this

# Suppress warnings and set pandas options
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load datasets
train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

# Combine datasets
combine = train._append(test, ignore_index=True)
print(f"Combined dataset shape: {combine.shape}")

# Define a function to remove patterns (e.g., @handles)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

# 1. Remove Twitter handles (@user)
combine['processed_tweet'] = np.vectorize(remove_pattern)(combine['tweet'], r"@[\w]*")

# 2. Remove punctuations, numbers, and special characters
combine['processed_tweet'] = combine['processed_tweet'].str.replace("[^a-zA-Z#]", " ", regex=True)

# 3. Remove smaller words
combine['processed_tweet'] = combine['processed_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# 4. Normalize text (lemmatization)
token = combine['processed_tweet'].apply(lambda x: x.split())
lemmatizer = WordNetLemmatizer()
combine['processed_tweet'] = token.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
combine['processed_tweet'] = combine['processed_tweet'].apply(lambda x: ' '.join([w for w in x]))

# Display the first 10 rows of the processed dataset
print(combine.head(10))
