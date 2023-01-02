"""
Description: Generates logistic regression models and their log_loss and accuracy scores. Helper file to classifier_analysis.

(Author: Kerria Pang-Naylor)
"""
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import pickle

def make_LogisticRegressionClassifier(x_train, y_train):
    """Helper to classifier_performance. Trains and returns  logistic regression model."""
    logreg = LogisticRegression(C=1e5)
    logreg.fit(x_train, y_train)
    return logreg

def logreg_accuracy(x_train,  x_test, y_train, y_test, logreg):
    """Helper to classifier_performance. Returns accuracy of model"""
    return logreg.score(x_train, y_train), logreg.score(x_test, y_test), #log_loss(x_test, y_test)

def gen_crossentropy(x_train,  x_test, y_train, y_test, logreg):
    """Helper to classifier_performance. Generates cross entropy for test and train data (y = true value, x = thing you plug into br)"""
    vfunc = np.vectorize(logreg.predict_proba)

    prob_train = vfunc(x_train)
    prob_test = vfunc(x_test)

    train_ce = log_loss(y_train, prob_train)
    test_ce = log_loss(y_test, prob_test)
    return train_ce, test_ce


def classifier_performance(clf_name, vec_name, max_features = 3000, test_ratio = 0.25, ngram_range:tuple = (1,2), bio_or_twe:str = "TWE", balance_method:str = "duplicate", 
                            stop_words_ignore = 'english', column_name:str = "text", cross_entropy = False, ret_vecreg = False):
    """
    Returns accuracy or cross entropy of logistic regression model.
        INPUTS: max_features (int, current number of features the model will use), test_ratio (float, test to train ratio), bio_or_twe
            (str, 'BIO' to use the bio classification model, 'TWE' to use the tweet classification model), ngram_range (tuple, range of n_grams considered)
            balance_method (str, 'duplicate' or 'delete'), stop_words_ignore ('english' to ignore stopwords, None to not ignore), column_name (str, name of 
            column that has text), cross_entropy (bool, True to measure log_loss, False to measure accuracy), ret_vecreg (bool, True to save vectorizer
            and regression model)
    """
    if bio_or_twe == "TWE":
        train_df = pd.read_pickle("classification-data/TWEETS-datasets/FULL_BOTH_TWEETS_CLASSIFICATION.pkl") # insert path to df with columns of "label" and "total" (column with clean text)
    elif bio_or_twe == "BIO":
        train_df = pd.read_pickle("classification-data/BIOS-datasets/FULL_BOTH_BIOS_CLASSIFICATION.pkl") # insert path to df with columns of "label" and "total" (column with clean text)

    # BALANCE THE DATA POINTS HERE (optional)
    difference  = abs(train_df["label"].value_counts()[1] - train_df["label"].value_counts()[0])
    if balance_method == "delete":
        drop_df = train_df[train_df["label"] == 0].sample(difference)
        drop_indices = list(drop_df.index)
        train_df = train_df[~train_df.index.isin(drop_indices)]
    elif balance_method == "duplicate":
        duplicates = train_df[train_df["label"] == 1].sample(n = difference)
        train_df = train_df.append(duplicates)
    
    # extract label column from train df to to the target var (y)
    targets = train_df['label'].values

    # Drop the 'label' column (has just been stored in targets)
    train_df.drop("label", axis = 1, inplace = True) # axis = 1 --> columnwise drop, inplace = edits own df

    # make tfidf transformer (transforms word count matrix to tf-idf representation (term frequency, inverse document frequency), rare words in only one type of documents are more important to classification for tfidf)
    transformer = TfidfTransformer(smooth_idf=False)
    count_vectorizer = CountVectorizer(stop_words = stop_words_ignore, ngram_range=ngram_range, min_df = 50, analyzer='word', max_features=max_features) #(ngrams being 1 to 2 means that count vectorizer considers only bigrams and unigrams)
    train_counts = count_vectorizer.fit_transform(train_df[column_name].values)
    train_tfidf = transformer.fit_transform(train_counts)
    
    x_train, x_test, y_train, y_test = train_test_split(train_tfidf, targets, random_state = 0, test_size = test_ratio) # x = input, y = label
    logreg = make_LogisticRegressionClassifier(x_train, y_train)

    if ret_vecreg:
        joblib.dump(logreg, clf_name)
        pickle.dump(count_vectorizer, open(vec_name, 'wb'))
        return logreg, count_vectorizer
    if cross_entropy:
        return gen_crossentropy(x_train,  x_test, y_train, y_test, logreg)
    else:
        return logreg_accuracy(x_train, x_test, y_train, y_test, logreg)

        

