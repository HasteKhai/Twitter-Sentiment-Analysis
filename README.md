# Twitter Sentiment Analysis

## Project Definition
Sentiment analysis is an application of Natural Language Processing (NLP) that uses methods and techniques for extracting subjective information from text or speech, such as opinions or attitudes. It involves classifying a piece of text as positive, negative, or neutral.

### Task
This NLP project aims to detect hate speeches in tweets. For simplciity, we assume that a tweet contains hate speech if it has a racist or sexist sentiment associated to it. Therefore, the goal is the classify racist or sexist tweets from others.

### Dataset
The provided datasets, a training set and a test set, contain tweets (around 32 000 instances). Here is an exhaustive list of features in the MECP inventory dataset.

<!-- Features taken from train_E6oV3lV.csv -->
> ### Features (Exhaustive List)
> - ID
> - Label
> - Tweet

## Model Design

### Class of Models
Supervised learning algorithms are suitable for SKU mapping because we have labeled data (MECP's inventory dataset) and want to predict the corresponding SKU for items in competitor datasets based on their features. For the SKU mapping segment of this project, a classification model is most appropriate, as it involves categorizing a competitor's item into one of MECP's items.


### Algorithm 1: K-NN
Using MECP's inventory dataset as the training dataset, we will feed entries from competitor's dataset and compute distance, find k-nearest point(s) from MECP's dataset to find out what the item is supposed to be mapped to. For nominal categorical features (ex: Type/ Tax Status) or Textual Data (ex: Name, Description, SKU) we would use the Jaccard Distance.

Because of the nature of the task, we believe that K=1. The reason being, since every point in the MECP dataset is a different item (no 2 points are same item), then the closest point would be what the model considers the answer.

In this case, we could use the SKU as the label(MECP uses this currently in their code).


### Algorithm 2: Random Forest Classification
During the training phase, the Random Forest Classification model will learn the relationships between the features of MECP's products and their respective identifiers, such as SKU. By training on this labeled dataset, the model gains the ability to discern patterns and associations between product attributes and identifiers.

In the prediction phase, entries from the competitor's dataset will be processed through the trained model. The model will analyze the features of these competitor products and predict the most probable corresponding item within MECP's inventory.

### Comparison
K-NN would be the better algorithm to use because of the nature of the dataset. Since every entry in MECP's dataset has a unique label (10 thousand unique SKUs), the tree would have a really high depth which would hinder the performance of RF. Furthermore, it would be hard to decide where to split the dataset for each branch since each label is unique (no information gain with splits), possibly leading the model to overfit.
