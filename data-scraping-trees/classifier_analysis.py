"""
Description: Run to display performance of logistic regression classifier with different varying n_gram range, max features, test ratios, etc
(Arbitrarily) chosen default values: n_gram_range = (1,2), max_features = 3000, balance_method = duplicate. Note that running this may
take a few minutes.
Creates graph of either log_loss or accuracy versus the number of features considered by the model

(Author: Kerria Pang-Naylor)
"""

# imports
from classifier_fnct import classifier_performance # import function that generates classification models
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# INPUTS/GLOBAL VARIABLES
log_loss = True # True if you want to graph cross-entropy, False if you want to graph accuracy
test_ratio = 0.25
ngram_range = (1,3)
balance_method = "duplicate" # duplicate or delete
stop_words_ignore = None # 'english' to ignore English stopwords, None to include stopwords
column_name = "text" # name of column with text data
bio_or_twe = "TWE" # "BIO" for Twitter bios or "TWE" for Tweets
step = 5 # step size between feature amount

# range of features tested
min_mfeatures = 800
max_mfeatures = 1500

title = f"{min_mfeatures}_to_{max_mfeatures}_features_{log_loss}_{ngram_range}_{balance_method}_{bio_or_twe}_{stop_words_ignore}_{step}"

# PREPARATION FUNCTS
"""Make pandas dataframe of how feature amount influences accuracy or cross-entropy"""

def make_max_featuresDF_accuracy(min_mfeatures = min_mfeatures, max_mfeatures = max_mfeatures, step = step, export:bool = False):
    """Measure accuracy with different values of max_mfeatures"""
    df = pd.DataFrame({"num_features": list(range(min_mfeatures, max_mfeatures+1, step))})
    df["temp"] = df["num_features"].apply(lambda num_f: classifier_performance(max_features = num_f, 
        test_ratio=test_ratio, ngram_range = ngram_range, balance_method=balance_method, 
        stop_words_ignore=stop_words_ignore, bio_or_twe=bio_or_twe, column_name=column_name))

    df["accuracy"] = df["temp"].apply(lambda x: x[0]) # test accuracy
    df["test_or_train"] = ["train"]*len(df)
    
    testdf = pd.DataFrame()
    testdf["num_features"] = df["num_features"]
    testdf["accuracy"] = df["temp"].apply(lambda x: x[1])
    testdf["test_or_train"] = ["test"]*len(df)

    df.drop("temp", axis = 1, inplace = True)

    combine = pd.concat([df, testdf])
    combine.to_pickle(f"classifier_perf_data/{title}.pkl")
    return combine

def make_max_featuresDF_CE(min_mfeatures = min_mfeatures, max_mfeatures = max_mfeatures, step = step, export:bool = False):
    """Measure cross-entropy with different values of max_mfeatures"""
    df = pd.DataFrame({"num_features": list(range(min_mfeatures, max_mfeatures+1, step))})
    df["temp"] = df["num_features"].apply(lambda num_f: classifier_performance(max_features = num_f, 
        test_ratio=test_ratio, ngram_range = ngram_range, balance_method=balance_method, 
        stop_words_ignore=stop_words_ignore, bio_or_twe=bio_or_twe, column_name=column_name, cross_entropy = True))
    df["log_loss"] = df["temp"].apply(lambda x: x[0])
    df["test_or_train"] = ["train"]*len(df)

    testdf = pd.DataFrame()
    testdf["num_features"] = df["num_features"]
    testdf["log_loss"] = df["temp"].apply(lambda x: x[1])
    testdf["test_or_train"] = ["test"]*len(df)

    df.drop("temp", axis = 1, inplace = True)

    combine = pd.concat([df, testdf])
    combine.to_pickle(f"classifier_perf_data/{title}_LL.pkl")

    return combine


# Graph log_loss or accuracy with respect to feature amount

try:
    if log_loss:
        df = pd.read_pickle(f"classifier_perf_data/{title}_LL.pkl")
    else:
        df = pd.read_pickle(f"classifier_perf_data/{title}.pkl")
    
except:
    if log_loss:
        df = make_max_featuresDF_CE()
    else:
        df = make_max_featuresDF_accuracy()
    df.to_pickle(f"classifier_perf_data/{title}.pkl")

if log_loss:
    sns.scatterplot(data=df, x = "num_features", y = "log_loss", hue = "test_or_train").set(title = title)
    plt.savefig(f"classifier_perf_data/graphs/LL/{title}_LL.png")
    plt.show()
    plt.close()
else:
    sns.scatterplot(data=df, x = "num_features", y = "accuracy", hue = "test_or_train").set(title = title)
    plt.savefig(f"classifier_perf_data/graphs/{title}.png")
    plt.show()
    plt.close()
