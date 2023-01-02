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
import pickle
import plotly.express as px
from classifier_fnct import classifier_performance

"""
https://nbviewer.org/github/PhilChodrow/PIC16A/blob/master/content/NLP/NLP_1.ipynb - fix issue

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

"""

def make_LogisticRegressionClassifier(x_train,  x_test, y_train, y_test):
    """ Best model for binary classification.
    """
    logreg = LogisticRegression(C=1e5,max_iter=1000) # here, I increased the max_iter value since I was getting a "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT."
                                                     # I am unsure of the significance of this
    logreg.fit(x_train, y_train)
    return logreg

def classifier_performance(clf_name, vec_name, max_features = 3000, test_ratio = 0.25, ngram_range:tuple = (1,2), bio_or_twe:str = "TWE", balance_method:str = "duplicate", 
                            stop_words_ignore = 'english', column_name:str = "text", ret_vecreg = False):
    """Returns accuracy and cross entropy of logistic regression model"""
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
    logreg = make_LogisticRegressionClassifier(x_train,  x_test, y_train, y_test)
    #return x_train,  x_test, y_train, y_test, logreg
    if ret_vecreg:
        #joblib.dump(logreg, f"better_perf_models/reg_{max_features}_{bio_or_twe}_{ngram_range}_{stop_words_ignore}.joblib")
        #pickle.dump(count_vectorizer, open(f"better_perf_models/vec_{max_features}_{bio_or_twe}_{ngram_range}_{stop_words_ignore}.pkl", 'wb'))
        joblib.dump(logreg, clf_name)
        pickle.dump(count_vectorizer, open(vec_name, 'wb'))
        return logreg, count_vectorizer
    

        
# INPUTS FOR FINDING RIGHT VECTORIZER
# GLOBAL VARS
log_loss = True
test_ratio = 0.25
ngram_range = (1,3)
balance_method = "duplicate" # duplicate or delete
stop_words_ignore = 'english' # 'english' or None
column_name = "text"
bio_or_twe = "TWE" #"BIO" or "TWE"
column_name = "text"
step = 5


min_mfeatures = 800
max_mfeatures = 1500

max_features = 925
#df = pd.read_pickle("ExtractedTweets.pkl")
#df = pd.read_csv("Corona_NLP_test.csv")
clf_filename = f"better_perf_models/reg_{max_features}_{bio_or_twe}_{ngram_range}_{stop_words_ignore}_testrat_{test_ratio}_{balance_method}.joblib" # 'classification-models/TWE_balancedbyDUP.joblib'
vec_filename = f"better_perf_models/vec_{max_features}_{bio_or_twe}_{ngram_range}_{stop_words_ignore}.pkl" # classification-models/TWE_balancedbyDUP_vec.pkl'

try: 
    clf = joblib.load(open(clf_filename, 'rb'))
    vectorizer = pickle.load(open(vec_filename, 'rb'))
    print(sd)
except:
    clf, vectorizer = classifier_performance(clf_filename, vec_filename, max_features = max_features, 
        test_ratio=test_ratio, ngram_range = ngram_range, balance_method=balance_method, 
        stop_words_ignore=stop_words_ignore, bio_or_twe=bio_or_twe, column_name=column_name, ret_vecreg= True)



def classify(text, clf, vectorizer, prob = True):

    #clf = joblib.load(open(clf_filename, 'rb'))
    #vectorizer = pickle.load(open(vec_filename, 'rb'))
    
    if prob:
        pred = clf.predict_proba(vectorizer.transform([text]))
        return pred[0]
    else:
        pred = clf.predict(vectorizer.transform([text]))
        if pred[0] == 0:
            return "pro-choice"
        else:
            return "pro-life"

def assign_score(text, clf, vectorizer):
    pred = classify(text, clf, vectorizer)
    return pred[1] - pred[0]

def make_scored_tweets(df, vectorizer, clf, text_column):
    df["Score"] = df[text_column].apply(lambda x: assign_score(x, clf, vectorizer)) 
    df.to_pickle(f"classifier_perf_data/scored_datasets/{dataset_name}_scored")

"""SCORE ASSIGN AND GRAPH TWEET OPINION DISTRIBUTION"""
dataset_name = "FULL_BOTH_TWEETS"#"ExtractedTweets"
full_dataset_name = "classification-data/TWEETS-datasets/FULL_BOTH_TWEETS.pkl"
x_axis = "Score"
text_column = "tweet_text"#"Tweet"
party_divider = "root_username" # "Party"

try:
    df = pd.read_pickle(f"classifier_perf_data/scored_datasets/{dataset_name}_scored")
except:
    df_base = pd.read_pickle(full_dataset_name)
    make_scored_tweets(df_base, vectorizer, clf, text_column)


df = pd.read_pickle(f"classifier_perf_data/scored_datasets/{dataset_name}_scored")

def view_hist(df, x_axis, nbins = 300, color = party_divider, log_x = False, log_y = False):
    """
    General histogram generating function data. Can choose dataframe and name of column that creates the x_axis.
    NOTE: do not set log_x to True, it doesn't work with this function.
        INPUTS: df (pandas dataframe), x_axis (string, name of column you want to plot), color (str, column that determines 
                colors of overlaying histogram), log_x (bool, make x-axis log), log_y (bool, make y-axis log).
        RETURNS: None
    """

    fig = px.histogram(df, x=x_axis, color=color, marginal="violin", # can be `box`, `violin`
                         hover_data=df.columns, nbins = nbins, log_y = log_y, log_x=log_x)
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    fig.show()

#view_hist(df, x_axis = x_axis)


#LR = clf
#X_train = pd.read_pickle("classification-data/TWEETS-datasets/FULL_BOTH_TWEETS_CLASSIFICATION.pkl")
#result_df = pd.DataFrame({"coef" : LR.coef_[0], "word" : X_train.columns})
