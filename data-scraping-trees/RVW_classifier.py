"""
Description:
Classifier model aimed to distinguish betweet pro-life or pro-choice tweets. I switched to using pickle files rather 
than csv's here. 

(Author: Kerria Pang-Naylor)

References: WomenWhoCode IntroNLP course day 5 
IDEAS TO MAKE MODEL BETTER: 
- switch to have it analyze tokenized_joined
- balance categories
- add more data
- add bag of words model
"""

from time import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import joblib

#  INPUTS (This classification model generating script will be turned into function eventually)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bio_or_twe = "TWE" # "TWE" # whether or not classification is on Twitter bios or tweets
column_name = "text" # "text" (uncleaned)  or "cleaned# cleaned text (in list form)
balance_method = "delete"#"duplicate" # either "duplicate" (preferable) or "delete" to correct unbalanced dataset.
                             # Duplicate creates copies of under-represented datapoints for model, delete simply deletes excess datapoints of the majority
filename = "bio_classifier" # file name of pickle classifier
minority = 1 # if you're balancing by duplicating, you must enter the minority label
test_ratio = 0.25 # percentage of datapoints that are for testing. The rest will be for test
export = False # to export classifier model with filename or not
download_confusion = True # download confusion matrix or not
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------



if bio_or_twe == "TWE":
    train_df = pd.read_pickle("classification-data/TWEETS-datasets/FULL_BOTH_TWEETS_CLASSIFICATION.pkl") # insert path to df with columns of "label" and "total" (column with clean text)
elif bio_or_twe == "BIO":
    train_df = pd.read_pickle("classification-data/TWEETS-datasets/FULL_BOTH_BIOS_CLASSIFICATION.pkl") # insert path to df with columns of "label" and "total" (column with clean text)


if column_name == "cleaned":
    name_text = "cleaned"
else:
    name_text = "uncleaned"

if balance_method == "delete":
    plot_name = f"{name_text}_balanced-del_{bio_or_twe}"
elif balance_method == "duplicate":
    plot_name = f"{name_text}_balanced-dup_{bio_or_twe}"
else:
    plot_name = f"{name_text}_unbalanced_{bio_or_twe}"

if bio_or_twe == "BIO":
    plot_file_path = "confusion-matrices/bios"
else:
    plot_file_path = "confusion-matrices/tweets"

# BALANCE THE DATA POINTS HERE (optional)

difference  = abs(train_df["label"].value_counts()[1] - train_df["label"].value_counts()[0])
if balance_method == "delete":
    drop_df = train_df[train_df["label"] == 0].sample(difference)
    drop_indices = list(drop_df.index)
    train_df = train_df[~train_df.index.isin(drop_indices)]
elif balance_method == "duplicate":
    duplicates = train_df[train_df["label"] == 1].sample(n = difference)
    train_df = train_df.append(duplicates)
# display balance between labels. Currently is slightly imbalanced to favor pro-choice tweets

def display_balance(df):
    """
    Displays bar chart of balance between pro-life and pro-choice users in the dataframe.
        INPUTS: df (pandas dataframe)
        RETURNS: none
    NOTE: since the duplicated values will be added to both the training and testing set, this may skew the accuracy score
    """
    rcParams["figure.figsize"] = 10,8
    sns.countplot(x = df["label"]).set(title = "train label balance")
    plt.show()


display_balance(train_df)

# extract label column from train df to to the target var (y)
targets = train_df['label'].values

# Drop the 'label' column (has just been stored in targets)
train_df.drop("label", axis = 1, inplace = True) # axis = 1 --> columnwise drop, inplace = edits own df

# make tfidf transformer (transforms word count matrix to tf-idf representation (term frequency, inverse document frequency), rare words in only one type of documents are more important to classification for tfidf)
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1,2)) #(ngrams being 1 to 2 means that count vectorizer considers only bigrams and unigrams)

# fit train data with pre-lemmatized and tokenized data
#train_counts = count_vectorizer.fit_transform(train_df["total"].values)

# do it with tokenized_join instead(?)
train_counts = count_vectorizer.fit_transform(train_df[column_name].values)

# fit ngrams count to tfidf transformaers
train_tfidf = transformer.fit_transform(train_counts)

# split df into train and test data. Use automatic distributation of 75% train and 25% test. 
# documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_tfidf, targets, random_state = 0, test_size = test_ratio) # x = input, y = label

"""Try out different classification sklearn models"""

def display_accuracy(model_name, train_score, test_score):
    print(f"Accuracy of {model_name} classifer on train set: {round(train_score,3)}")
    print(f"Accuracy of {model_name} classifer on test set:  {round(test_score,3)}\n")

def make_ExtraTreesClassifier(x_train, x_test, y_train, y_test):
    """
    SMALLER SET
    Accuracy of ExtraTrees classifer on train set: 0.9921833598304186
    Accuracy of ExtraTrees classifer on test set:  0.6856120826709062
    """
    Extr = ExtraTreesClassifier(n_estimators = 5, n_jobs = 4) # n_estimators = how many trees, n_jobs =  # runs to fit the model
    Extr.fit(x_train, y_train)
    display_accuracy("ExtraTrees", Extr.score(x_train, y_train), Extr.score(x_test, y_test))
    
    return Extr
#make_ExtraTreesClassifier(x_train, x_test, y_train, y_test)

def make_AdaBoostClassifier(x_train, x_test, y_train, y_test):
    """
    SMALLER SET
    Accuracy of ExtraTrees classifer on train set: 0.683
    Accuracy of ExtraTrees classifer on test set:  0.659
    """
    Adab= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
    Adab.fit(x_train, y_train)
    display_accuracy("ExtraTrees", Adab.score(x_train, y_train), Adab.score(x_test, y_test))

    return Adab

def make_RandomForestClassifier(x_train, x_test, y_train, y_test):
    """
    SMALLER SET
    Accuracy of ExtraTrees classifer on train set: 0.964
    Accuracy of ExtraTrees classifer on test set:  0.667
    """
    RandomFC= RandomForestClassifier(n_estimators=5)
    RandomFC.fit(x_train, y_train)
    display_accuracy("ExtraTrees", RandomFC.score(x_train, y_train), RandomFC.score(x_test, y_test))

    return RandomFC

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
def make_NaiveBayesClassifier(x_train, x_test, y_train, y_test):
    """ 
    This one has done 2nd to best. Apparently NB is good for certain classification tasks where nodes aren't dependent on eachother (?)
    Accuracy of Naive Bayes classifer on train set: 0.966
    Accuracy of Naive Bayes classifer on test set:  0.771
    """
    NB = MultinomialNB()
    NB.fit(x_train, y_train)    

    display_accuracy("Naive Bayes",NB.score(x_train, y_train), NB.score(x_test, y_test))

    return NB


"""
---------------------------------------------------------------
    USE LOGISTIC REGRESSION FOR BEST BINARY CLASSIFICATION
---------------------------------------------------------------
"""
def make_LogisticRegressionClassifier(x_train,  x_test, y_train, y_test, filename = filename, export = export):
    """ Best model for binary classification.
    Accuracy of Naive Bayes classifer on train set: 0.992
    Accuracy of Naive Bayes classifer on test set:  0.78
    """
    logreg = LogisticRegression(C=1e5)
    logreg.fit(x_train, y_train)
    display_accuracy("Logistic Regression",logreg.score(x_train, y_train), logreg.score(x_test, y_test))

    if export:
        joblib.dump(logreg, f"classification-data/{filename}.joblib")
    # load file like joblib.load(filename)

    return logreg

logreg = make_LogisticRegressionClassifier(x_train, x_test, y_train, y_test)

# FOR LOGREG: Get accuracy (# correct predictions/total number of data points)
from sklearn.metrics import accuracy_score

predictions = logreg.predict(x_test)
score = accuracy_score(y_test,predictions)
print(f"Accuracy: {round(score*100,2)}%")

pred_prob = logreg.predict_proba(x_test)


# rn accuracy is 77.98%

# Make Confusion Matrix
from sklearn import metrics
CM = metrics.confusion_matrix(y_test, predictions)
print(CM)

plt.figure(figsize=(9,9))
sns_plot = sns.heatmap(CM, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
fig = sns_plot.get_figure()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(f"{plot_name}\n{all_sample_title}", size = 15)
plt.show()

print(plot_name)
if download_confusion:
    fig.savefig(f"{plot_file_path}/{plot_name}.png", dpi = 100)

