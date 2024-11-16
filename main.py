import re
import nltk
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # WordNet lemmatizer requires this

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
tokens = combine['processed_tweet'].apply(lambda x: x.split())

lemmatizer = WordNetLemmatizer()
combine['processed_tweet'] = tokens.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
combine['processed_tweet'] = combine['processed_tweet'].apply(lambda x: ' '.join([w for w in x]))

# Display the first 10 rows of the processed dataset
print(combine.head(10))

# WordCloud Plot
all_words = ' '.join([text for text in combine['processed_tweet']])
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500).generate(all_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

# Non racist/sexist words
# positive_words = ' '.join([text for text in combine['processed_tweet'][combine['label'] == 0]])
# wordcloud = WordCloud(width=800, height=500).generate(positive_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

# Racist/sexist words
negative_words = ' '.join([text for text in combine['processed_tweet'][combine['label'] == 1]])


# wordcloud = WordCloud(width=800, height=500).generate(negative_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

def hashtag_extract(tweets):
    return [re.findall(r"#(\w+)", tweet) for tweet in tweets]


positive_hashtags = hashtag_extract(combine['processed_tweet'][combine['label'] == 0])
negative_hashtags = hashtag_extract(combine['processed_tweet'][combine['label'] == 1])
positive_hashtags = sum(positive_hashtags, [])
negative_hashtags = sum(negative_hashtags, [])

# a = nltk.FreqDist(positive_hashtags)
# d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})
# d = d.nlargest(columns="Count", n=20)
# plt.figure(figsize=(16, 8))
# ax = sns.barplot(data=d, x="Hashtag", y="Count", palette='pastel')
# ax.tick_params(axis='x', rotation=45)
# ax.set(ylabel='Count')
# plt.show()

# b = nltk.FreqDist(negative_hashtags)
# e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# e = e.nlargest(columns="Count", n=20)
# plt.figure(figsize=(16, 8))
# ax = sns.barplot(data=e, x="Hashtag", y="Count", palette='pastel')
# ax.tick_params(axis='x', rotation=45)
# ax.set(ylabel='Count')
# plt.show()

# Feature Conversion

# Bag-of-Words Features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combine['processed_tweet'])
print(bow.shape)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combine['processed_tweet'])
print(tfidf.shape)

# Word2Vec Features
word2vec = gensim.models.Word2Vec(tokens, vector_size=200, window=5, sg=1, hs=0, negative=10, workers=2, seed=34)
word2vec.train(tokens, total_examples=len(combine['processed_tweet']), epochs=20)
print(word2vec.wv.most_similar(positive="dinner"))
print(word2vec.wv.most_similar(positive="trump"))


# Prepping Vectors for Tweets

# Create a vector for each tweet by taking the average of the vector of the words present in the tweet
def word_vector(twt, model, size):
    # Get vectors for valid words and compute the mean
    vectors = np.array([model.wv[word] for word in twt if word in model.wv])
    return vectors.mean(axis=0) if vectors.size > 0 else np.zeros(size)


# Compute word vectors for each tweet
wordvec_arrays = np.array([word_vector(tweet, word2vec, 200) for tweet in tokens])

# Convert to DataFrame
wordvec_df = pd.DataFrame(wordvec_arrays)
print(wordvec_df.shape)

# Doc2Vec Embedding
from tqdm import tqdm

tqdm.pandas(desc="progess-bar")
from gensim.models.doc2vec import TaggedDocument


def add_label(twt):
    return [TaggedDocument(words=s, tags=[f"tweet_{i}"]) for i, s in enumerate(twt)]


labeled_tweets = add_label(tokens)
print(labeled_tweets[:6])

doc2vec = gensim.models.Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=5, negative=7, min_count=5, workers=3,
                                alpha=0.1, seed=23)
doc2vec.build_vocab([i for i in tqdm(labeled_tweets)])
doc2vec.train(labeled_tweets, total_examples=len(combine['processed_tweet']), epochs=20)

docvec_arrays = np.array([doc2vec.dv[i] for i in range(len(combine))])
docvec_df = pd.DataFrame(docvec_arrays)
print(docvec_arrays.shape)
