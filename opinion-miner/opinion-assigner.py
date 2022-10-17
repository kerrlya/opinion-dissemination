"""
opinion-assigner.py : given binary classification model and dataset, will assign both continuous/
probabilistic and binary label to each datapoint.

(Author: Kerria Pang-Naylor)
"""
# import statements
import sys, joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split

#---MANUAL STUFF--- (Ignore)

# import classification models (doesn't work right now)
#sys.path.append('opinion-dissemination/data-scraping-trees/classification-models')
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1,2))

column_name = "text"

# new input data
bio_df = pd.read_pickle("datasets/bios.pkl")
tweet_df = pd.read_pickle("datasets/tweets.pkl")

targets_bio = bio_df['label'].values
targets_tweet = tweet_df['label'].values

bio_counts = count_vectorizer.fit_transform(bio_df[column_name].values)
tweet_counts = count_vectorizer.fit_transform(tweet_df[column_name].values)


# fit ngrams count to tfidf transformaers
train_tfidf_bio = transformer.fit_transform(bio_counts)
train_tfidf_tweet = transformer.fit_transform(tweet_counts)

# load pre-trained model
bio_model = joblib.load(open('classification-models/bio_classifier.joblib', 'rb'))
tweet_model = joblib.load(open('classification-models/tweet_classifier.joblib', 'rb'))

# get list of predictions
pred_bio = bio_model.predict(train_tfidf_bio).tolist()
pred_prob_bio = bio_model.predict_proba(train_tfidf_bio).tolist()

prob_0_bio = list(map(lambda x: x[0], pred_prob_bio))
prob_1_bio = list(map(lambda x: x[1], pred_prob_bio))


pred_tweet = tweet_model.predict(train_tfidf_tweet).tolist()
pred_prob_tweet = tweet_model.predict_proba(train_tfidf_tweet).tolist()

prob_0_tweet = list(map(lambda x: x[0], pred_prob_tweet))
prob_1_tweet = list(map(lambda x: x[1], pred_prob_tweet))

# add predictions as columns
bio_df["gen_label"] = pred_bio
bio_df["prob_0"] = prob_0_bio
bio_df["prob_1"] = prob_1_bio
bio_df["prob_pair"] = pred_prob_bio

tweet_df["gen_label"] = pred_tweet
tweet_df["prob_0"] = prob_0_tweet
tweet_df["prob_1"] = prob_1_tweet
tweet_df["prob_pair"] = pred_prob_tweet


bio_df.to_pickle("datasets/bios.pkl")
tweet_df.to_pickle("datasets/tweets.pkl")



