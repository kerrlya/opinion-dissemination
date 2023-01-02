"""Assigns continuous probabilistic opinion-scores to tweet datasets. Labeled with column of 'score'"""


"""INPUTS"""
display_dist = True # where or not to display distribution chart of labeled dataframe
custom_clf_path = None
custom_vec_path = None
original_data_path = "external-datasets/SurgeAI_bios.pickle" # or "external-datasets/COVID19_tweets.pkl"

text_column = "bio"#" "text" if tweets, "bio" if bios
party_divider = "Category"#"Category" # column name that denotes original label

"""Model generation global variables"""
log_loss = True
test_ratio = 0.25
ngram_range = (1,3)
balance_method = "duplicate" # duplicate or delete
stop_words_ignore = 'english' # 'english' or None
bio_or_twe = "BIO" #"BIO" or "TWE"
column_name = "text"
step = 5

max_features = 925

general_filename = f"{max_features}_{bio_or_twe}_{ngram_range}_{stop_words_ignore}_testrat_{test_ratio}_{balance_method}"
clf_filename = f"better_perf_models/clf_{general_filename}.joblib" # or 'classification-data/classification-models/TWE_balancedbyDUP.joblib'
vec_filename = f"better_perf_models/vec_{max_features}_{bio_or_twe}_{ngram_range}_{stop_words_ignore}.pkl" # "opinion-miner/classification-models/TWE_balancedbyDUP_vec.pkl"


# imports
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

try: 
    clf = joblib.load(open(clf_filename, 'rb'))
    vectorizer = pickle.load(open(vec_filename, 'rb'))
except:
    clf, vectorizer = classifier_performance(clf_name = clf_filename, vec_name = vec_filename, max_features = max_features, 
        test_ratio=test_ratio, ngram_range = ngram_range, balance_method=balance_method, 
        stop_words_ignore=stop_words_ignore, bio_or_twe=bio_or_twe, column_name=column_name, ret_vecreg= True)


"Assign scores"

x_axis = "Score"

def classify(text, clf, vectorizer, prob = True):    
    """
    Helper to assign_score. Returns categorical classification or probabilistic classification.
        INPUTS: text (str, any string in English), clf (scikitlearn classifier), vectorizer (scikitlearn vectorizer),
            prob (bool, True to return probabilistic prediction)
    """
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
    """
    Helper to make_scored_tweets, returns subtract score of probablistic prediction.
        INPUTS: text (str, any string in English), clf (scikitlearn classifier), vectorizer (scikitlearn vectorizer),
                prob (bool, True to return probabilistic prediction)
    """
    pred = classify(text, clf, vectorizer)
    return pred[1] - pred[0]

def make_scored_tweets(df, vectorizer, clf, text_column):
    """
    Adds new column 'Score' for subtract score of tweet.
        INPUTS: df (Pandas dataframe with tweets/text), clf (scikitlearn classifier), vectorizer (scikitlearn vectorizer),
                text_column (str, name of column in df that has text)
    """
    df["Score"] = df[text_column].apply(lambda x: assign_score(x, clf, vectorizer)) 
    return df
    #df.to_pickle(f"classifier_perf_data/scored_datasets/{dataset_name}_scored")

df_base = pd.read_pickle(original_data_path)
df_base= df_base.dropna(subset="bio")
df = make_scored_tweets(df_base, vectorizer, clf, text_column) # pd.read_pickle(f"classifier_perf_data/scored_datasets/{dataset_name}_scored")

def view_hist(df, x_axis, nbins = 150, color = party_divider, log_x = False, log_y = False):
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

view_hist(df, x_axis = x_axis)
print("Standard deviation: ", np.std(df["Score"]))

#LR = clf
#X_train = pd.read_pickle("classification-data/TWEETS-datasets/FULL_BOTH_TWEETS_CLASSIFICATION.pkl")
#result_df = pd.DataFrame({"coef" : LR.coef_[0], "word" : X_train.columns})
