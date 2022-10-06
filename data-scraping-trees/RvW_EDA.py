"""
Description:
Exploratory Data Analysis and plots on language and popularity distribution of Roe v. Wade tweets.

(Author: Kerria Pang-Naylor)
"""

from turtle import color
from sklearn.manifold import trustworthiness
from NLP_preprocessing import *
from matplotlib import pyplot as plt
import plotly.express as px
import ast
import math
import numpy as np


# df = pd.read_csv("RVW_EDA.csv")
# df_learn = pd.read_csv("RvW_preprocess.csv")
# df2 = pd.read_csv("RVW_FULL.csv")


def view_hist(df, x_axis, nbins = 200, color = "root_username", log_x = False, log_y = False):
    """
    General histogram generating function for roe v. wade data. Can choose dataframe and name of column that creates the x_axis.
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


def plot_top_stopwords(df, num = 15, column = "lower"):
    """
    Plots num most common stop words in histogram.
        INPUTS: df (pandas dataframe), num (int, number of top common words listed), column (str, name of column consinsting 
        of a list of words for each row (str)),
        RETURNS: top num stopwords (list)
    """
    from nltk.corpus import stopwords
    stopwords = list(stopwords.words('english'))
    text_lists = list(df[column])
    
    text = []
    for text_L in text_lists:
        text += text_L
   

    stop_dic = {}

    for word in text:
        if word in stopwords:
            if word in stop_dic:
                stop_dic[word] += 1 
            else:
                stop_dic[word] = 1 
    
    top = sorted(stop_dic.items(), key=lambda x:x[1],reverse=True)[:num] # list of most common words
    print(top)
    #unzip 'top'
    x, y = zip(*top)
    plt.figure(figsize=(num,num))
    plt.title(f"{num} Most Common Stopwords")
    plt.bar(x,y)
    plt.show()
    return top

def exclude(word):
    """
    Optional helper to plot_top_nonstopwords. Sees if word inputted is in the list is in exclude_these or if a port of 
    the word is in any of the exclude_these.
    
    """
    exclude_these = ['abortion','women','right','people','life','states','rights',"scotus",'wade',
 'v',
 'roevwade',
 'get',
 'court',
 'state',
 'would',
 'us',
 'supreme',
 'overturned',
 'one',
 'decision',
 'want',
 'today',
 'need',
 'like',
 'care',
 'know',
 'going'] # 20 most common words from both camps


    for word_ in exclude_these:
        if word_ in word:
            return True
    
    return False

def plot_top_nonstopwords(df, side = "Both", num = 20, title = "None",label_column = "label", excludeTop = True, column_name = "final_cleaned", is_pickle = True):
    """
    Plots num most common stop words in histogram for side's tweets (NARAL or march for life), if excludeTop, the excludes 
    the top 10 most common words from both camps.
        INPUTS: df (pandas dataframe that contains column of "root" that contains), side (str, either "Both", "NARAL", or "March_for_life"),
            num (int, number of top non stopwords that are included), label_column (str, the column name that contains labels) excludeTop (bool, whether or not to call exclude 
            function/exclude certain words), column_name (name of column that contains list of words), is_pickle (bool)
        RETURNS: list of num top non-stopwards
    """

    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))

    if side == "Both":
        print("remain")
        df_ = df
    else:
        df_ = df.loc[df[label_column] == side]

    #print(df)
    text_lists = list(df_[f"{column_name}"])
    #print(text_lists)
    text = []
    for text_L in text_lists:
        if not is_pickle:
            text += ast.literal_eval(text_L)
        else:
            text += text_L

    #print(text_lists[:2])
    stop_dic = {}


    for word in text:
        if word not in stopwords:
            if excludeTop:
                if exclude(word):
                    continue

            if word in stop_dic:
                stop_dic[word] += 1 
            else:
                stop_dic[word] = 1 
    
    #print(stop_dic)
    
    top=sorted(stop_dic.items(), key=lambda x:x[1],reverse=True)[:num]

    #unzip 'top'
    plt.rcParams.update({'font.size':7})
    x,y = zip(*top)
    plt.figure(figsize=(num, num))
    plt.bar(x,y)
    plt.title(title, fontsize = 15)
    plt.show()

    return top

def make_buckets(nbins, range_):
    """Helper to graph_surface_of_hist. Imports number of bins (nbins, int), hist_data (list), and range_ (tuple, the minimum and maximum x-values) to create a list 
    of tuple pairs of each 'bin' range for the histogram. Returns dictionary template with the correct bucket ranges as keys (all values initially set to 0)."""

    bin_width = math.ceil((range_[1] - range_[0])/nbins) # must round up to nearest integer value
    indices = list(range(range_[0],range_[1], bin_width)) # these are the 'tic marks' for each histogram bin start or stop

    if indices[-1] < range_[1]: # if last value of indices is less than max x value, add max x value to indices
        indices.append(range_[1])

    buckets = {(indices[i], indices[i+1]) : 0 for i in range(len(indices)-1)} # create tuple pairs for each histogram bucket
    return buckets
def get_coords_dict(nbins, hist_data,range_):
    """Helper to graph_surface_of_hist. Imports number of bins (nbins, int), hist_data (list), and range_ (tuple, the minimum and maximum x-values) to create a list 
    of tuple pairs of each 'bin' range for the histogram. Returns dictionary with the correct bucket ranges as keys and runs through hist_data to add counts for each
    range."""
    coord_dict = make_buckets(nbins, range_)

    for val in hist_data: # fill in histogram dictionary
        for bucket in coord_dict:
            if val >= bucket[0] and val < bucket[1]:
                coord_dict[bucket] += 1

    return coord_dict

def get_coords(nbins, hist_data,range_):
    """Helper to graph_surface_of_hist. Imports number of bins (nbins, int), hist_data (list), and range_ (tuple, the minimum and maximum x-values) to create (1) a 
    dictionary of buckets with counts (representing a histogram), and then averaging the range of each bin to get the x_coordinate, while the number of counts becomes 
    the y coordinate. Returns list of x and corresponding y coordinates """

    coord_dict = get_coords_dict(nbins, hist_data, range_)

    x_buckets = list(coord_dict.keys())
    y = list(coord_dict.values())

    # average bucket ranges

    x = [bin[0] + (bin[1] - bin[0])/2 for bin in x_buckets]

    return x, y 

def to_log(L):
    """Helper to graph_surface_of_hist. Inputs list of integers L. Simply turn all 0 L terms to 1 (log(1) = 0), then take log 10 of every term in L"""
    L = [term if term != 0 else 1 for term in L] # turn zeroes to ones
    L_log = [math.log10(term) for term in L]

    return L_log

     
def graph_surface_of_hist(df, nbins = 200, range_ = (0,1000), column_name = "likes+retweets",log_x = True, log_y = True, scatter = True, title = "Histogram Curve of Number of Followers", show_fit = True):
    """
    Inputs dataframe with column_name. Creates graph of the curve of the top of the histogram (of retweets and likes of dataframe). Outputs deg 1 line of best fit.
        INPUTS: df (pandas dataframe), nbins (# of bins of histogram), range_ (tuple of minimum and maximum numbers to be considered in histogram),
                column_name (str, column name of column containing integers that the histogram will be based off of), log_x (bool, if plotted with x-axis on log scale )
                log_y (bool, if plotted with y-axis on log scale), scatter (bool, whether or not to display scatter plot as well), title (str, title of graphs), plot_fit (bool, 
                whether to plot line of best fit)
        RETURNS: fit_str (string of y = mx + b line of best fit)
    """
    hist_data = list(df[(df[column_name] >= range_[0]) & (df[column_name] <= range_[1])][column_name]) # get all tweets within the range

    x, y = get_coords(nbins, hist_data, range_)
    
    x_label = f"x (num {column_name})"
    y_label = "y (counts)"

    if log_x:
        x_og = x
        x = to_log(x)
        x_label = f"log10({column_name})"
    if log_y:
        y_og = y
        y = to_log(y)
        y_label = "log10(counts)"

    m, b = np.polyfit(x,y,1)
    
    fit_str_rounded = f"y = {round(m,4)}x + {round(b,4)}"
    
    fit_str = f"y = {m}x + {b}"
    fit_x = list(range(math.floor(min(x)), math.ceil(max(x)) +  1) )
    fit_y = [m*x_ + b for x_ in fit_x]

    plt.plot(x,y, label = "Data-based curve")
    if show_fit:
        plt.plot(fit_x, fit_y, "r",label = f"Line of best fit {fit_str_rounded}")
    plt.legend()

    plt.title(f"{title}")


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    if scatter:
        plt.scatter(x,y)
        plt.title(f"{title} (scatter)")
        if show_fit:
            plt.plot(fit_x, fit_y, "r", label = f"Line of best fit {fit_str_rounded}")

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    if log_x and log_y: # show alpha curve calculated for double log graph
        
        plt.plot(x_og, y_og, label = 'Original data-based curve')
        x_ = x_og
        y_ = [(x_term**m)*(10**b) for x_term in x_]

        plt.plot(x_, y_, "r", label = f'Estimated curve')
        plt.ylabel("counts")
        plt.xlabel(f"num {column_name}")
        plt.legend()
        plt.show()
    elif log_y:
        plt.plot(x, y_og, label = 'Original data-based curve')
        x_ = x
        y_ = [10**(x_term*m + b) for x_term in x_]
        plt.plot(x_,y_,label = "Estimated curve")
        plt.ylabel("counts")
        plt.xlabel(f"num {column_name}")
        plt.legend()
        plt.show()
    print(f"Line of best fit: {fit_str}")

    return fit_str

def balance_by_del(df, minority = 1):
    """Quick function that balances the labels in a dataframe by deletion. Copied from RVW_classifier
    """

    difference  = abs(df["label"].value_counts()[1] - df["label"].value_counts()[0])
    drop_df = df[df["label"] == 0].sample(difference)
    drop_indices = list(drop_df.index)
    df = df[~df.index.isin(drop_indices)]
    return df

df = pd.read_pickle("classification-data/FULL_BOTH_TWEETS_NODUP_FOLLOWINGINFO.pkl")
df2 = pd.read_pickle("classification-data/FULL_BOTH_TWEETS.pkl")
df2 = df2[df2["retweets+likes"] < 1000]
#df2 = balance_by_del(df2)