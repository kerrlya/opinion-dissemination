import tweepy
import config_tweepy
import numpy as np
import pandas as pd
from pandas import DataFrame
import json # converts strings of things to the actual things, ex:  bruh = json.loads("[1,2,3]")
import networkx as nx
import matplotlib.pyplot as plt

"""
DISCONTINUED VERSION OF DFGEN. PLEASE LOOK AT DFGEN TO GENERATE TWITTER DATAFRAMES.

"""
client = tweepy.Client(bearer_token = config_tweepy.BEARER_TOKEN)
IDs = ["1537170742764240900", "1537150818901577728", "1537166889754865664","1537239733876985856", "1535143667979104256"]

testdf = pd.read_csv

#RETWEET CASCADE NETWORK DATA FROM TWEEPY
def genCascadeNetworkData(id_L, filename, sameNetwork = False, followerDegree = 0):
    """Returns csv with information about retweets of certain tweetID(s). This tweetID's should ideally be tweets from within the same underlying network
            INPUTS: 
                id_L - list of tweet IDs whose retweets you want to scrape
                filename - (str) name of csv to export (no need to add .csv)
                sameNetwork - booleon that tells whether or not tweets in id_L a
                followerDegree - (int) if using same network, the number of repetitions outward a follower network will be generated
            OUTPUT: excel (or csv) with categories of 

            Search is currently limited to 10 things (i.e. user who've liked)
            """
    cascade_df = pd.DataFrame() # cascade dataframe


    #cascade_df categories:
    tweetIDs = id_L # list of tweet ID's whose data and retweets you want to scrape
    tweeterID = [] # author of tweet
    tweetText = [] # text in the tweet
    retweeters = [] # (list of lists) list of user IDs of those who have retweeted each tweet; later find way to find when something was retweeted
    quote_retweets = [] # (list of dictionaries) quote retweet IDs to quote retweet text
    likers = [] # (list of lists) users who have just liked the tweet
    likersAndRetweeters = [] # (list of lists) users who have liked and retweeted
    tweetTime = [] # time created for each tweet, convert this to epoch time later since its easier
    
    

    # follower_df categories
    ### TBD


    for ID in id_L:
        tweet = client.get_tweet(ID, tweet_fields = ['created_at', 'text'])
        tweetText += [tweet.data.text]
        tweetTime += [str(tweet.data.created_at)]

        #Create list of liker IDs
        likers_here = client.get_liking_users(id = ID)
        likers_here_L = []
        for liker in likers_here.data:
            likers_here_L += [liker.id]
        likers += [likers_here_L] # appends list as term to likers list

        #create retweeters and liker and retweeter IDs
        retweeters_here = client.get_retweeters(id = ID)
        retweeters_here_L = []
        likersAndRetweeters_here_L = []

        for retweeter in retweeters_here.data:
            retweeters_here_L += [retweeter.id]

            if retweeter.id in likers_here_L:
                likersAndRetweeters_here_L += [retweeter.id]
        
        retweeters += [retweeters_here_L]
        likersAndRetweeters += [likersAndRetweeters_here_L]

        #create quote retweets dictionary for each tweet
        quote_retweets_here = client.get_quote_tweets(id = ID, tweet_fields = ['created_at', 'text'])
        quote_retweets_D = {}

        for quote_retweet in quote_retweets_here.data:
            quote_retweets_D[quote_retweet.id] = [str(quote_retweet.created_at), quote_retweet.text]
        
        quote_retweets += [quote_retweets_D]

        # create new entry for each quote retweet along wiht their retweets
        

    # now put lists into dataframe
    cascade_df["Tweet IDs"] = tweetIDs
    cascade_df["Tweet Text"] = tweetText
    cascade_df["Retweeter IDs"] = retweeters
    cascade_df["Quote Retweets (id: [time, text])"] = quote_retweets
    cascade_df["Liker IDs"] = likers
    cascade_df["Liker and Retweeter IDs"] = likersAndRetweeters
    cascade_df["Time of Tweet"] = tweetTime

    # export as excel
    cascade_df.to_excel(f'{filename}_cascade.xlsx')



#genCascadeNetworkData(IDs,"firstData")



#RETWEET CASCADE NETWORKX DIGRAPH FROM
def genCascade(filename):
    """Returns networkX digraph of retweets of a certain tweet. Limits retweets to those that have liked and retweeted.
        """
    cascade_df = pd.read_excel(f'{filename}')
    for index, row in cascade_df.iterrows():
        start_node = row['Tweet IDs']
        graph = nx.DiGraph()
        
        retweets = json.loads(row["Retweeter IDs"])
        for retweet in retweets:
            graph.add_edge(start_node,retweet)

        nx.draw(graph, with_labels = True)
        plt.show()

genCascade("firstData.xlsx")


    
    

