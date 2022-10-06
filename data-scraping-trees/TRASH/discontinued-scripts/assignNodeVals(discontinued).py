from cmath import inf
import nltk

import tweepy
from textblob import TextBlob
import config_tweepy
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import graph_dataframe_generator

from node_classes import user

client = tweepy.Client(bearer_token = config_tweepy.BEARER_TOKEN, wait_on_rate_limit=True)

auth = tweepy.OAuthHandler(config_tweepy.CONSUMER_KEY, config_tweepy.CONSUMER_SECRET) # setup api v1
auth.set_access_token( config_tweepy.ACCESS_TOKEN,  config_tweepy.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

aoc = user(username = "AOC") # test case

"""OLD VERSION OF DATA PRE-PROCESSING. SEE NLP_preprocessing FOR NEW VERSION"""

"""Train data with large sets from anti and pro abortion ppl on twitter binary labeling, combine with subjectivity and polarity in vector/use them to scale opinion scalar"""

        
def cleanTxt(text):
    """Cleans tweets for textBlob analysis"""
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # get rid of tags ('r' tells python it's a raw string)
    text = re.sub(r'#', '', text) # removes the # symbol in front of hashtags
    text = re.sub(r'RT[\s]+', '', text) # removes 'RT' for retweets
    text = re.sub(r'https?:\/\/\S+', '', text) # Removes hyperlinks question mark --> can have 0 or 1 s's

    return text

def getSubjectivity(text):
    """Returns (0,1) subjectivity score for tweet text. Subjectivity quantifies the amount of personal opinion 
    and factual information contained in the text. The higher subjectivity means that the text contains personal 
    opinion rather than factual information.  Polarity lies between [-1,1], -1 defines a negative sentiment and 
    1 defines a positive sentiment."""
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(text):
    return TextBlob(text)

def correctEmojis(text):
    pass

# def displayWordCloud(column):
#     """Displays word cloud for text list/column"""
#     allWords = ' '.join( [twts for twts in column] )
#     wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size= 119).generate(allWords)
#     plt.imshow(wordCloud, interpolation= "bilinear")
#     plt.axis('off')
#     plt.show()

def getTweetScore(text, inSupportPos = True):
    """Rudimentary polarity based tweet score"""
    if inSupportPos == True:
        return getPolarity(text)*5
    else:
        return (-1)*getPolarity(text)*5



#filename = "newAOCDF_cascade.xlsx"
def addScoresToDF(filename):
    df = pd.read_excel(f'{filename}.xlsx')
    df["Clean Tweets"] = df["text"].apply(cleanTxt)
    df["Subjectivity"] = df["Clean Tweets"].apply(getSubjectivity)
    df["Polarity"] = df["Clean Tweets"].apply(getPolarity)
    df["Score"] = df["text"].apply(getTweetScore)

    df.to_excel(f'{filename}_update.xlsx')

def newAddScoresToDF(filename):
    df = pd.read_excel(f'{filename}.xlsx')

    df["Clean Tweets"] = df["text"].apply(cleanTxt)
    df["Subjectivity"] = df["Clean Tweets"].apply(getSubjectivity)
    df["Polarity"] = df["Clean Tweets"].apply(getPolarity)
    df["Old Score"] = -df["text"].apply(getTweetScore)

    dems = list(pd.read_excel("democrats.xlsx")["Username"])
    reps = list(pd.read_excel("republicans.xlsx")["Username"])

    affiliation = []
    numRepFollowing = []
    numDemFollowing = []


    df["Discrete score/Affiliation"] = [np.nan]*len(df)
    df["Num Dems Following"] = [np.nan]*len(df)
    df["Num Reps Following"] = [np.nan]*len(df)

    num_index = 0

    for index, row in df.iterrows():
        user_ = user(username = row["username"])
        aff = user_.demOrRep()
        df.loc[num_index, "Discrete score/Affiliation"] = aff
        df.loc[num_index, "Num Dems Following"] = user_.num_dem_following
        df.loc[num_index,"Num Reps Following"] = user_.num_rep_following
        df.to_excel(f'{filename}_update.xlsx')

        num_index += 1


    # df["Discrete score/Affiliation"] = affiliation
    # df["Num Dems Following"] = numDemFollowing
    # df["Num Reps Following"] = numRepFollowing
    

        


    df.to_excel(f'{filename}_update.xlsx')

#newAddScoresToDF("NewNodeAssignTest")

bruh = user(username = "AOC")
query = '(#RoevWade OR #RoeVWade OR #roevwade OR "unborn" OR "abortion" OR (Roe Wade)) -BREAKING -"check out" -podcast -news -is:retweet -is:quote lang:en -is:reply -has:media -has:links'#is:verified roevwade OR #roevswade -is:retweet -has:media lang:en -is:reply' # -is:nullcast', won't let me filter out ads w/o better license (#scotus OR #SCOTUS OR SCOTUS) 
#https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query

def scrapeRow(tweet):
    """Imports tweet object, returns row list"""
    user = client.get_user(id = tweet.author_id)
    return [tweet.id, user.data.username, tweet.text, tweet.author_id, tweet.public_metrics["like_count"], tweet.public_metrics["retweet_count"], tweet.public_metrics["reply_count"], tweet.public_metrics["quote_count"]]


def scrapeRelevantTweets(filename, query, num_results = 10):
    """Created df of relevant tweets"""    

    tweets = client.search_recent_tweets(query = query, tweet_fields = ["author_id", "text", "public_metrics", 'created_at', 'context_annotations'], max_results = num_results)
    row_0 = tweets.data[0]
    user = client.get_user(id = row_0.author_id)
    df = pd.DataFrame({'tweet_id': [row_0.id], 'username': [user.data.username], 'text': [row_0.text], 'author_id': [row_0.author_id], 'num_likes': [row_0.public_metrics["like_count"]], 'num_retweets': [row_0.public_metrics["retweet_count"]],'num_replies': [row_0.public_metrics["reply_count"]], 'num_quotes': [row_0.public_metrics["quote_count"]]})
    
    for tweet in tweets.data:
        df.loc[len(df.index)] = scrapeRow(tweet)
    # for index, row in df.iterrows():
    #     df["author_id"] = 
    
    df.to_excel(f'{filename}.xlsx')
    return df

#user keywords "unborn" pro-life

def getRelevantLikedTweets(neg_opinion_L, pos_opinion_L, keywords):

    pass

def scrapeTrainingData(filename, neg_opinion_L, pos_opinion_L, keywords):
    """Scrapes training data for gauging whether a tweet is for or against a certain sentiment where you know the positions of popular users.
    Tweets by people who have liked those posts will adopt their binary opinion state (i.e., for/against roeVwade overturn)"""

    pass

def manualScoreAssign(userID, neg_opinion_L,pos_opinion_L, keywords):
    """Uses (1) whether or not a user is following/liking certain popular users to determine sign of opinion, and (2) scaling that by how frequently they like and/or (re)tweet certain hashtags or keywords"""
    pass



def getPoliticiansLists():
    """opens politicians dataset, picks out things you want, converts to pickle file and excel file"""
    df = pd.read_csv('dataset.csv')
    dem_d = {"Name": [], "Username": []}
    rep_d = {"Name": [], "Username": []}

    new_d = {"Name": [], "Username": [], "Party": []} #, "Followers" : [], "Following" : [], "Tweets": [], "Listed": []}    # dictionary for new_df
    for index, row in df.iterrows():
        try:

            # first correct POTUS to joe biden
            if row["Twitter_username"] == "POTUS":
                dem_d["Name"].append("Joe Biden")
                dem_d["Username"].append(row["Twitter_username"])

                new_d["Username"].append(row["Twitter_username"])
                new_d["Name"].append("Joe Biden")
                new_d["Party"].append("Democrat")
            
            else:

                # info = basicUserInfo(row["Twitter_username"])
                if "Democratic" in row["Political_party"]:
                    dem_d["Name"].append(row["Name"])
                    dem_d["Username"].append(row["Twitter_username"])
                    
                    new_d["Username"].append(row["Twitter_username"])
                    new_d["Name"].append(row["Name"])
                    new_d["Party"].append("Democrat")
                    #new_d["ID"].append(row["Account_ID"])

                    # new_d["Followers"].append(info["followers"])
                    # new_d["Following"].append(info["following"])
                    # new_d["Tweets"].append(info["tweet_count"])
                    # new_d["Listed"].append(info["listed_count"])
                
                if "Republican" in row["Political_party"]:
                    rep_d["Name"].append(row["Name"])
                    rep_d["Username"].append(row["Twitter_username"])

                    new_d["Name"].append(row["Name"])
                    new_d["Username"].append(row["Twitter_username"])
                    new_d["Party"].append("Republican")
                    #ew_d["ID"].append(row["Account_ID"])
                    
                    # new_d["Followers"].append(info["followers"])
                    # new_d["Following"].append(info["following"])
                    # new_d["Tweets"].append(info["tweet_count"])
                    # new_d["Listed"].append(info["listed_count"])
        except:
            continue

    new_df = pd.DataFrame(new_d)
    dem_df = pd.DataFrame(dem_d)
    rep_df = pd.DataFrame(rep_d)

    dem_df.to_excel("democrats.xlsx")
    rep_df.to_excel("republicans.xlsx")
    new_df.to_excel("newPolData.xlsx")

    #return new_df, dem_L, rep_L

aoc = user(username = "AOC")
#aoc_following = aoc.getFollowingList()

# keywords:
# pro-choice: #womensrightsarehumanrights, #clarencethomasmustgo, #mybodymychoice, #voteblue, #abortionrightsarehumanrights
