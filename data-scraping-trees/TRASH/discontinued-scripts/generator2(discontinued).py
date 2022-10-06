import tweepy
import config_tweepy
import numpy as np
import pandas as pd
from pandas import DataFrame
import json # converts strings of things to the actual things, ex:  bruh = json.loads("[1,2,3]")
import networkx as nx
import matplotlib.pyplot as plt
import ast

"""
DISCONTINUED VERSION OF DFGEN. PLEASE LOOK AT DFGEN TO GENERATE TWITTER DATAFRAMES.
"""

# ast.literal_eval(insert string her) can evaluate strings of things bettwer
"""Possible client.get_tweet() parameters [attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics,public_metrics,referenced_tweets,reply_settings,source,text,withheld]"""

client = tweepy.Client(bearer_token = config_tweepy.BEARER_TOKEN, wait_on_rate_limit=True)

auth = tweepy.OAuthHandler(config_tweepy.CONSUMER_KEY, config_tweepy.CONSUMER_SECRET)
auth.set_access_token(config_tweepy.ACCESS_TOKEN, config_tweepy.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth) #call the API; this is API v1, which requires elevated access or above :(

# test cases
IDs = ["1537170742764240900", "1537150818901577728", "1537166889754865664","1537239733876985856", "1535143667979104256"]
#newID = "1538991118603411461"
#newID = "1538986794703781888"
#userID = "44196397"

userID = "1251638956359385089"# add new user ID here for covid conspiracy thing
tweetID = "1540755940542119938" #AOC's tweet
ID = "138203134" #AOC's user ID
testdf = pd.read_csv
quoteID = "1541110057231867908"
#functions!!!

def isValidUser(username = None, userID = None, tweetID = None):
    """Returns True is userID belongs to valid user or if user ID of tweetID is a valid user (must input either userID or tweetID
        if user has more than. Perform filter with this function on followingDF generator"""
        # can't do with just essential access since I need to user tweepy v1:((
    if userID != None:
        user = client.get_user(id = userID, user_fields = ["public_metrics","created_at","url","entities"] )
    elif username != None:
        user = client.get_user(username = username, user_fields = ["public_metrics","created_at","url","entities"])
    else:
        response = client.get_tweet(id = tweetID, expansions = ['author_id'])
        userID = str(response.data.author_id)
        #print(userID)
        user = client.get_user(id = userID, user_fields = ["public_metrics","created_at","url","entities"] )

    followers_count = user.data.public_metrics["followers_count"]
    following_count = user.data.public_metrics["following_count"]
    tweet_count = user.data.public_metrics["tweet_count"]
    
    if followers_count < 20:
        return False
    elif following_count < 20:
        return False
    elif tweet_count < 10:
        return False
    else:
        return True


def genTweetInfo(ID, root_ID = None):
    """Returns information about inputted tweet ID as a long tuple with  (username, tweetID (str), tweetText (list), retweeters (list), 
    quote_retweets (dict), likers (list), likersAndRetweeters (list), tweetTime)"""

    #get created_at time and text of tweet
    tweet = client.get_tweet(ID, tweet_fields = ['created_at', 'text', 'author_id', 'public_metrics', "referenced_tweets", 'lang']) # eventually use these direct fields instead
    tweetText = tweet.data.text
    tweetTime = str(tweet.data.created_at)
    authorID = tweet.data.author_id
    #root_ID = tweet.data.referenced_tweets
    

    tweeter = client.get_user(id = authorID)
    username = tweeter.data.username

    # Get list of liker IDs
    likers_here = client.get_liking_users(id = ID)
    likers_here_L = []
    try:
        for liker in likers_here.data:
            likers_here_L += [liker.id]
        likers = list(filter(lambda userID: isValidUser(userID = userID), likers_here_L)) # appends list as term to likers list
    except:
        likers = np.nan

    #create retweeters and liker and retweeter IDs
    retweeters_here = client.get_retweeters(id = ID)
    retweeters_here_L = []
    likersAndRetweeters_here_L = []    

    try:
        for retweeter in retweeters_here.data:
            retweeters_here_L += [retweeter.id]

            if retweeter.id in likers:
                likersAndRetweeters_here_L += [retweeter.id]
                    
        retweeters = retweeters_here_L
        likersAndRetweeters = list(filter(lambda id: isValidUser(userID = id), likersAndRetweeters_here_L))
    except:
        retweeters = np.nan
        likersAndRetweeters = np.nan
    
    #create quote retweets dictionary for each tweet
    #quote_retweets_here = client.get_quote_tweets(id = ID, max_results = 100, tweet_fields = ['created_at', 'text']) # with essential access can't do more than 10 requests
    quote_retweets_here = client.get_quote_tweets(id = ID, tweet_fields = ['created_at', 'text',"in_reply_to_user_id","lang"])
    quote_retweets_D = {}

    try:    
        for quote_retweet in quote_retweets_here.data:
            if isValidUser(tweetID = quote_retweet.id):
                quote_retweets_D[quote_retweet.id] = [str(quote_retweet.created_at), quote_retweet.text]
            
                if quote_retweet.id in likers_here_L:
                    likersAndRetweeters += [quote_retweet.id]
        
        quoteRetweets = quote_retweets_D
        

    except:
        quoteRetweets = np.nan # can also get root_ID from in_reply_to_user_id 


    
    num_retweets = tweet.data.public_metrics['retweet_count']
    num_replies = tweet.data.public_metrics['reply_count']
    num_likes = tweet.data.public_metrics['like_count']
    num_quotes = tweet.data.public_metrics['quote_count']
    # root_ID = str(tweet.data.in_reply_to_user_id)
    # print(root_ID)

    if root_ID == None:
        root_ID = np.nan
    #     liked_root = np.nan
    #     follows_root =np.nan
    # else:
    #     if ID in likers_here_L:
    #         liked_root = True
    #     if ID 
    # CANT ADD liked_root and follows_root booleon BECAUSE ITS TOO EXPENSIVE TO CALL AND SEARCH THROUGH ALL THE LISTS OF LIKERS AND FOLLOWERS
    
    return [username, str(ID), authorID, tweetText, retweeters, quoteRetweets, likers, likersAndRetweeters, tweetTime, root_ID, num_retweets, num_replies, num_likes, num_quotes] #, liked_root, follows_root]


def genUserInfo(userID, max_f = 30):
    """Creates row for followership excel."""
    response_root = client.get_user(id = userID, user_fields = ['public_metrics'])
    username_root = response_root.data.username
    num_followers = response_root.data.public_metrics["followers_count"]
    num_following = response_root.data.public_metrics["following_count"]
    tweet_count = response_root.data.public_metrics["tweet_count"]
    listed_count = response_root.data.public_metrics["listed_count"]


    response = client.get_users_followers(id = userID, max_results = max_f)
    followers = {}
    try:
        for entry in response[0]:
            if isValidUser(userID=entry.id):
                followers[str(entry.username)] =  entry.id
        if len(followers) == 0:
            followers = np.nan
    except:
        followers = np.nan
    
    
    response = client.get_users_following(id = userID, max_results = max_f)
    following = {}
    try:  #filter for only valid followers
        for entry in response[0]:
            if isValidUser(userID = entry.id):
                following[entry.username] =  entry.id
        if len(following) == 0:
            following == np.nan
    except:
        following = np.nan
    
    #now get additional info


    return [username_root, userID, followers, following, num_followers, num_following, tweet_count, listed_count]



def genFollowershipDF(ID,filename, max1 = 10, max2 = 10, baseFollowing = True):
    """Generates followership Data Frame. max1 is inner count of followees (max number of followers we consider when adding list of folloewrs). Max2 is outer count: max number of followers period"""
    follower_df = pd.DataFrame()
    root = genUserInfo(ID)
    follower_df["Usernames"] =[root[0]]
    follower_df["IDs"] = [root[1]]
    follower_df["Followers"] = [root[2]]
    follower_df["Following"] = [root[3]] 
    follower_df["# followers"] = [root[4]]
    follower_df["# following"] = [root[5]]
    follower_df["# tweets"] = [root[6]]
    follower_df["# listed"] = [root[7]]


    count2 = 0
    for index, row in follower_df.iterrows():
        id_here = row["IDs"]

        count = 0
        if not baseFollowing:
            for follower in row["Followers"]:
                #print(follower)  
                follower_id = row["Followers"][follower]
                follower_df.loc[len(follower_df.index)] = genUserInfo(str(follower_id))
                count+= 1
            
                if count > max1: # eventually change to cycles outward with followerships
                    break
        else:
            for friend in row["Following"]:
                #print(follower)  
                friend_id = row["Following"][friend] #row[] is a dictionary rn
                follower_df.loc[len(follower_df.index)] = genUserInfo(str(friend_id))
                count+= 1
            
                if count > max1: # eventually change to cycles outward with followerships
                    break

        count2 +=1

        if count2> max2:
            break
        
    follower_df.to_excel(f'{filename}.xlsx')
        
    return follower_df


def genCascadeDF(ID, filename):
    """Returns dataframe with information about retweets of certain tweetID(s). This tweetID's should ideally be tweets from within the same underlying network
            INPUTS: 
                id - tweet ID whose retweets you want to scrape
                filename - (str) name of csv to export (no need to add .csv)
                sameNetwork - booleon that tells whether or not tweets in id_L a
                followerDegree - (int) if using same network, the number of repetitions outward a follower network will be generated
            OUTPUT: excel (or csv) with categories of 

            Search is currently limited to 10 things (i.e. user who've liked)
            """
    
    cascade_df = pd.DataFrame() # cascade dataframe for this root tweet'
    username_, ID_, authorID_, tweetText_, retweeters_, quoteRetweets_, likers_, likersAndRetweeters_,tweetTime_, root_ID, num_retweets, num_replies, num_likes, num_quotes = genTweetInfo(ID) # get first row for original tweet
#[username, str(ID), authorID, tweetText, retweeters, quoteRetweets, likers, likersAndRetweeters, tweetTime, root_ID, num_retweets, num_replies, num_likes, num_quotes]
    cascade_df["Usernames"] = [username_]
    cascade_df["IDs"] = [ID]
    cascade_df["Author"] = [authorID_]
    cascade_df["Text"] = [tweetText_]
    cascade_df["Retweeter IDs"] = [retweeters_]
    cascade_df["Quote Retweets"] = [quoteRetweets_]
    cascade_df["Liker IDs"] = [likers_]
    cascade_df["Liker and Retweeter IDs"] = [likersAndRetweeters_]
    cascade_df["Time of Tweet"] = [tweetTime_]
    cascade_df["Root tweet_id"] = [root_ID]
    cascade_df["# retweets"] = [num_retweets]
    cascade_df["# replies"] = [num_replies]
    cascade_df["# likes"] = [num_likes]
    cascade_df["# quotes"] = [num_quotes]
    #cascade_df["liked_root"] = [np.nan]

    for index, row in cascade_df.iterrows():
        id_here = row["IDs"]

        for quote_ID in row["Quote Retweets"]:
            cascade_df.loc[len(cascade_df.index)] = genTweetInfo(str(quote_ID), id_here)

    # export cascade as excel
    cascade_df.to_excel(f'{filename}_cascade.xlsx')

    return cascade_df


#RETWEET CASCADE NETWORK DATA FROM TWEEPY/ THIS IS THE OLD BAD VERSION
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
    
    dfs = [] # list of data frames

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

    for ID in id_L:

        cascade_df = pd.DataFrame() # cascade dataframe for each original retweet

        tweet = client.get_tweet(ID, tweet_fields = ['created_at', 'text'])
        tweetText += [tweet.data.text]
        tweetTime += [str(tweet.data.created_at)]

        #Create list of liker IDs
        likers_here = client.get_liking_users(id = ID)
        likers_here_L = []
        for liker in likers_here.data:
            likers_here_L += [liker.id]
        likers += [likers_here_L] # appends list as term to likers list
        print(likers)

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

            if quote_retweet.id in likers_here_L:
                likersAndRetweeters += [quote_retweet.id]
        
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

    print(cascade_df)
    return cascade_df    

def outConnect(startUser, connectL, graph):
    for outNode in connectL:
        graph.add_edge(startUser, outNode)

def inConnect(endUser, connectL, graph):
    for inNode in connectL:
        graph.add_edge(inNode, endUser)


#RETWEET CASCADE NETWORKX DIGRAPH FROM
def genCascade(filename, includeRetweets = False, onlyLikes = True):
    """Returns networkX digraph of retweets of a certain tweet. Limits retweets to those that have liked and retweeted.
        if onlyLikes is true, only include those who've retweeted and liked """
    cascade_df = pd.read_excel(f'{filename}')
    graph = nx.DiGraph()

    # connect quote tweets
    outConnect(cascade_df['Usernames'][0], cascade_df["Usernames"][1:], graph)

    #now connect all retweets
    if includeRetweets:
        for index, row in cascade_df.iterrows():
            start_node = row['Usernames']        
            retweeter_IDs = row["Retweeter IDs"]

            if type(retweeter_IDs) == str:
                retweeter_IDs = json.loads(retweeter_IDs)

                retweeters = []
                for ID in retweeter_IDs:
                    tweeter = client.get_user(id = ID)
                    retweeters += [tweeter.data.username + "(R)"]
                outConnect(start_node, retweeters, graph)
    
    if onlyLikes:
        for index, row in cascade_df.iterrows():
            start_node = row['Usernames']        
            retweeter_IDs = row["Liker and Retweeter IDs"]
            
            if type(retweeter_IDs) == str:
                retweeter_IDs = json.loads(retweeter_IDs)

                retweeters = []
                for ID in retweeter_IDs:
                    tweeter = client.get_user(id = ID)
                    retweeters += [tweeter.data.username + "(L&R)"]
                outConnect(start_node, retweeters, graph)
    

    nx.draw(graph, with_labels = True)
    plt.show()

def genFollowershipNet(filename, includeFollowing = True):
    df = pd.read_excel(f'{filename}')
    graph = nx.MultiDiGraph()

    for index, row in df.iterrows():
        root = row["Usernames"]
        #print("\n" + root)
        

        try:
            followers = ast.literal_eval(row["Followers"])
            following = ast.literal_eval(row["Following"])
            #print("followers: ", followers)
            #print("following: ", following)

            outConnect(root,list(followers),graph)
            inConnect(root,list(following), graph)

        except:
            continue
    

    nx.draw(graph, with_labels = True)
    plt.show()

    return graph


#genCascade("firstData.xlsx")
#cascade_df = genCascadeDF(newID, "dfTestNewNew")

#genTweetInfo(newID)

#genCascade("dfTestNewNew_cascade.xlsx", False)
#genCascade("dfTestNewNew_cascade.xlsx")

#genFollowershipDF(userID, "test")
#genFollowershipNet("test_followerships.xlsx")

#genFollowershipDF(userID, "larger_followerDF2",10,4)
#G = genFollowershipNet("larger_followerDF2_followerships.xlsx")