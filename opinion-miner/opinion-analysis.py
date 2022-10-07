""""""
import plotly.express as px
import pandas as pd

def view_hist(df, x_axis, nbins = 300, color = "label", log_x = False, log_y = False):
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




bio_df = pd.read_pickle("datasets/bios.pkl")
tweet_df = pd.read_pickle("datasets/tweets.pkl")

bio_df["subtract_score"] = bio_df["prob_pair"].apply(lambda x: x[1] - x[0])
tweet_df["subtract_score"] = tweet_df["prob_pair"].apply(lambda x: x[1] - x[0])


view_hist(bio_df, "subtract_score", log_y = True)

