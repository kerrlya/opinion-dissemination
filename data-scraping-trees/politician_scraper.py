from tweet_scraper import scrape_followers_of
import tweet_scraper as ts
import pandas as pd
import numpy as np
"""Generates Labeled CSVs of followers of politicians """
def try_get_followers(username):
    try:
        return scrape_followers_of(username)
    except:
        return np.nan

pol_df = pd.read_pickle("dem-rep/politicians.pkl").head(5)
followers_D = {}

df = pd.DataFrame()
df["Username"] = pol_df["Username"]
df["Followers"] = df["Username"].apply(try_get_followers)

df.to_pickle("Politicians_followers.pkl")






