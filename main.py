import re
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')


pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

###Sample of non racist/sexist tweets
#print(train[train['label'] == 0].head(10))
###Sample of racit/sexist tweets
#print(train[train['label'] == 1].head(10))
###Dimensions of datasets
#print('Train set dimensions: ', train.shape, '| Test set dimensions: ', test.shape)
###Label-distribution
#print(train["label"].value_counts())

###In the train dataset, we have 7% of tweets labeled as racist/sexist, and 93% of tweets labeled as non racist/sexist.
###We notice that there is a class imbalance.

###Distribution of length of the tweets
# length_train = train['tweet'].str.len()
# length_test = test['tweet'].str.len()
# plt.hist(length_train, bins=20, label='train_tweets')
# plt.hist(length_test, bins=20, label='test_tweets')
# plt.legend()
# plt.show()

### Cleaning the raw text data by removing noise such as punctuation, special characters, nubmers, and terms that don't
### carry much weight in context.

combine = train._append(test, ignore_index=True)
print(combine.shape)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'', input_txt)
    return input_txt

###1. Remove twitter handles (@user).
combine['processsed_tweet'] = np.vectorize(remove_pattern)(combine['tweet'], r"@[\w]*")
###2. Remove punctuations, numbers, and special characters.
combine['processsed_tweet'] = combine['processsed_tweet'].str.replace("[^a-zA-Z#]"," ", regex=True)
###3. Remove smaller words.
combine['processsed_tweet'] = combine['processsed_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
###4. Normalize text data. For example, reduce loves, loving, and lovable to love (base word).
lemmatizer = WordNetLemmatizer()
combine['processsed_tweet'] = combine['processsed_tweet'].apply(word_tokenize)
combine['processed_tweet'] = combine['processed_tweet'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
print(combine.head(10))
