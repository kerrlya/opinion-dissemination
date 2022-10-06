"""
Description:
This script contains many newer tweet scraping tools, specifically generating labeled dataframes for data analysis, natural 
language processing, and classification.

(Author: Kerria Pang-Naylor)
"""
import tweepy
import pandas as pd
from twitter_user_class import *
import config_tweepy
import numpy as np
from functools import reduce
import time
import random

client = tweepy.Client(bearer_token = config_tweepy.BEARER_TOKEN, wait_on_rate_limit=True)

auth = tweepy.OAuthHandler(config_tweepy.CONSUMER_KEY, config_tweepy.CONSUMER_SECRET) # setup api v1
auth.set_access_token( config_tweepy.ACCESS_TOKEN,  config_tweepy.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def scrape_followers_of(root_username = None, rootID = None):
    """Gets list 10000 followers from root_username
            INPUTS: root_username (str) OR rootID (int or str)
            OUTPUT: followers (list of followers)    
    """
    time.sleep(random.randomint(1,4))
    user_root = user(id = rootID, username =root_username)
    followers = user_root.getFollowersList() # gets up to 130,000 follwers
    return followers


def split_list(l, n):
    """Helper to build_tweet_pkl. Helper to get_lists_users. Splits list into lists of list with length of n.
        INPUTS: l (list), n (int)"""
    for i in range(0,len(l),n):
        yield l[i:(i+n)]


def get_lists_users(users):
    """Helper to build_tweet_pkl. Returns query form of users in list with 'from:'
            INPUTS: users (list of users)
            OUTPUTS: list of users with 'from:' appended to each """
    users_with_from = [f"from:{user}" for user in users]
    users_L = list(split_list(users_with_from, 40)) # lists of 
    return users_L

def build_query_from_users(users_LL):
    """Helper to build_tweet_pkl. Takes one 100 term users_LL list, returns joined string that will become twitter search 
    query."""
    last_index = len(users_LL) - 1
    query_users = ""
    for i, user in enumerate(list(users_LL)):
        if i != last_index:
            query_users += f"{user} OR "
        else:
            query_users += str(user)
    
    return query_users

def query_list(users_L, query):
    """Helper to build_tweet_pkl. Builds list of querys from a user list and twitter query specification"""
    query_L = [f"({build_query_from_users(users_LL)}) {query}" for users_LL in users_L]
    return query_L

def get_mentions(tweet):
    """Input tweet object (with entities). Extract list of usernames/user_ids mentioned in tweet. Not currently used."""
    mentions_L = []
    
    try: # in case the mention_dict_L is empty
        mention_dict_L = tweet.entities["mentions"]
        for mention_dict in mention_dict_L:
            mentions_L.append((mention_dict["username"], mention_dict["id"]))
    except Exception:
        return np.nan
    return mentions_L

# time range for scraping RvW tweets
start_time = "2022-06-24T00:00:00Z"
end_time = "2022-07-3T00:00:00Z" 

def build_tweet_section(query, start_time, end_time):
    """Helper to build_tweet_excel, builds a series of lists to be appended to each column, returns small dataframe to be concatenated into larger df"""
    
    info = {"tweet_id" : [], "author_id" : [], "created_at" : [], "retweet_count" : [], "reply_count" : [], "like_count" : [], "quote_count" : [], "tweet_text" : [], "conversation_id" : [], "in_reply_to_user_id" : [], "mentions":[], "context_annotations" : []}

    try:
        req = client.search_all_tweets(query, max_results = 100,start_time = start_time, end_time = end_time, place_fields = ["full_name", "name", "place_type"], tweet_fields = ["author_id", "conversation_id","created_at", "in_reply_to_user_id", "public_metrics","entities", "context_annotations"])
        time.sleep(random.randint(1,3))

        for tweet in req.data:
            info["tweet_id"].append(tweet.id)
            info["author_id"].append(tweet.author_id)
            info["created_at"].append(str(tweet.created_at))
            info["retweet_count"].append(tweet.public_metrics["retweet_count"])
            info["reply_count"].append(tweet.public_metrics["reply_count"])
            info["like_count"].append(tweet.public_metrics["like_count"])
            info["quote_count"].append(tweet.public_metrics["quote_count"])
            info["tweet_text"].append(tweet.text)

            info["conversation_id"].append(tweet.conversation_id)
            info["in_reply_to_user_id"].append(tweet.in_reply_to_user_id)
            
            
            info["mentions"].append(get_mentions(tweet))
            
            try:
                info["context_annotations"].append(tweet.context_annotations)
            except Exception:
                print("no context annotations")
                info["context_annotations"].append(np.nan)

    except Exception:
        print("Something skipped")
        pass

    return info


def build_tweet_pkl(users, twitter_query, *, filename = None, root_username = None,start_time = None, end_time = None):
    """
    Inputs list of users creates pickle file with tweet information.
        INPUTS: users (list of usernames)
    """
    df_dict = {"tweet_id" : [], "author_id" : [], "created_at" : [], "retweet_count" : [], "reply_count" : [], "like_count" : [], "quote_count" : [], "tweet_text" : [], "conversation_id" : [], "in_reply_to_user_id" : [], "mentions" : [], "context_annotations" : []}
    #df_dict = {"tweet_id" : [], "author_id" : [],"tweet_text" : []}

    users_L = get_lists_users(users) # splits users into a list of 100 term lists
    query_L = query_list(users_L, twitter_query) # puts 'from:' in front of each usernames

    for query in query_L:
        section = build_tweet_section(query, start_time= start_time, end_time=end_time)
        df_dict["tweet_id"].extend(section["tweet_id"])
        df_dict["author_id"].extend(section["author_id"])
        df_dict["created_at"].extend(section["created_at"])
        df_dict["retweet_count"].extend(section["retweet_count"])
        df_dict["reply_count"].extend(section["reply_count"])
        df_dict["like_count"].extend(section["like_count"])
        df_dict["quote_count"].extend(section["quote_count"])
        df_dict["tweet_text"].extend(section["tweet_text"])
        df_dict["conversation_id"].extend(section["conversation_id"])
        df_dict["in_reply_to_user_id"].extend(section["in_reply_to_user_id"])
        df_dict["mentions"].extend(section["mentions"])
        df_dict["context_annotations"].extend(section["context_annotations"])
        
        df = pd.DataFrame(df_dict)
        df.to_pickle(f"{filename}.pkl") # upload to csv each time just in case something weird happens
        
    #dfs = [build_tweet_section(query) for query in query_L[:2]]
    #final_df = pd.concat(dfs)
    try:
        if root_username:
            df["root_username"] = len(df)*[root_username]
        
        if filename:
            df.to_pickle(f"{filename}.pkl")
    except Exception:
        print("exporting didn't work")

    return df


def filter_valid_users(user_L):
    """Inputs list of username, outputs list of those who are not protected and active (i.e., are "valid") """

    filtered_L = list(filter(lambda user_: user(user_).is_valid_user(), user_L))
    return filtered_L


def get_full_followership_csv(user_L, filename):
    """
    Generates csv with with user_L along the indices and the first 5000 user's they are following as a list on the next column.
    
    INPUTS:
        :param user_L: List of users whose follower's you'd like to scrape.
        :type user_L: List
        :param filename: name of csv file you will upload
        :type filename: string
    RETURNS:
        :return: dataframe of csv that has been uploaded
        :rtype: pandas dataframe
    """

    df = pd.DataFrame({"user": user_L})

    def temp(username):
        try:
            return user(username = username).getFollowingList()
        except Exception:
            return "invalid user"

    df["following_list"] = df["user"].apply(temp)

    df.to_csv(f"{filename}.csv")
    # THIS TOOK WAYYYYYY TOO LONG. USE PREVIOUS METHOD INSTEAD
    return df

def get_users_bios(usernames = None, user_ids = None):
    """Helper to get_full_users_bios Takes in list of up to 100 usernames or user_ids. Returns their bios."""

    req = client.get_users(ids = user_ids, usernames = usernames, user_fields = ["description"])
    bios = [user_.description for user_ in req.data ]
    print("worked")
    return bios

def get_full_users_lists(usernames = None, user_ids = None):
    """Helper to get_full_users_bios. Takes in list of any number of usernames or user_ids. Returns series of 100 term lists with usernames."""

    if usernames != None:
        usernames = list(dict.fromkeys(usernames)) # get rid of duplicates
        users = usernames
    else:
        user_ids = list(dict.fromkeys(user_ids))
        users = user_ids
  
    indices = list(range(len(users)))[::100]

    user_lists = []

    for index in indices:
        user_list = users[index:(index+100)]
        user_lists.append(user_list)

    return user_lists

def flatten_list_of_lists(LL):
    """Helper to get_full_users_bios. Flattens list of lists to flat list."""
    flat_list = []

    for L in LL:
        flat_list += L
    
    return flat_list

def get_full_users_bios(id_list):
    """
    Gets full list bios for twitter user IDs in id_list. Returns lists of bios and corresponding ids
        INPUT: id_list (list of user IDs)
        RETURNS: bios_L (list of bios (str)), id_lists_flat (list of corresponding users in bios_L)
    """

    id_lists = get_full_users_lists(id_list)

    bios_L = []
    
    for user_list in id_lists:
        time.sleep(random.randint(2,4))
        #print(user_list)
        bios = get_users_bios(user_ids = user_list)
        #print(bios)
        bios_L += bios

    id_lists_flat = flatten_list_of_lists(id_lists)
    return bios_L, id_lists_flat
        
"""
----------------------------
    Some scraping code
----------------------------   
"""

'''Possible query:
    (#RoevWade OR #RoeVWade OR #roevwade OR "unborn" OR "abortion" OR (Roe Wade)) -BREAKING -"check out" -podcast -news -is:retweet -is:quote lang:en -is:reply -has:media -has:links -nullcast'''
twitter_query = "(#RoevWade OR unborn OR abortion OR (Roe Wade)) -is:retweet -is:quote lang:en -is:reply -has:media -nullcast -has:links"



#  CODE FOR CREATING LARGE TWITTER DATASETS BASED ON FOLLOWERS OF A USER AND A QUERY

# start_time = "2022-06-24T00:00:00Z"
# end_time = "2022-07-4T00:00:00Z"
# #test_req = client.search_all_tweets(query_L[0], max_results = 100,start_time = "2022-06-24T00:00:00Z", end_time = "2022-07-3T00:00:00Z", place_fields = ["full_name", "name", "place_type"], tweet_fields = ["context_annotations", "author_id", "conversation_id","created_at", "in_reply_to_user_id", "public_metrics"] )

# life_followers  = list(pd.read_pickle("RVW_March_for_Life.pkl")["March_for_life"])
# choice_followers  = list(pd.read_pickle("RVW_NARAL.pkl")["NARAL"])

# twitter_query = "(#RoevWade OR unborn OR abortion OR (Roe Wade)) -is:retweet -is:quote lang:en -is:reply -has:media -nullcast -has:links"

#life_df = build_tweet_pkl(users = life_followers, twitter_query=twitter_query, filename = "FULL_LIFE_TWEETS", root_username = "March_for_Life", start_time=start_time, end_time=end_time)

#choice_df = build_tweet_pkl(users = choice_followers, twitter_query=twitter_query, filename = "FULL_CHOICE_TWEETS", root_username = "NARAL", start_time=start_time, end_time=end_time)

# CODE FOR CREATING TWITTER DATASETS FOR CLASSIFICATION - scraping twitter bios
# df_choice = pd.read_pickle("classification-data/FULL_CHOICE_TWEETS.pkl")
# df_life = pd.read_pickle("classification-data/FULL_LIFE_TWEETS.pkl")


# choice_users = list(df_choice["author_id"])
# life_users = list(df_life["author_id"])



# choice_bios, choice_users = get_full_users_bios(id_list = choice_users)
# life_bios, life_users = get_full_users_bios(id_list = life_users)

# try:
#     choice_bios_df = pd.DataFrame({"IDs": choice_users, "NARAL bios" : choice_bios})
#     choice_bios_df.to_pickle("classification-data/FULL_CHOICE_BIOS.pkl")

#     life_bios_df = pd.DataFrame({"IDs" : life_users, "March_for_life bios" : life_bios})
#     life_bios_df.to_pickle("classification-data/FULL_LIFE_BIOS.pkl")
# except Exception:
#     pass

# try:
#     df_life.to_pickle("FULL_LIFE_TWEETS_bios.pkl")
#     df_choice.to_pickle("FULL_CHOICE_TWEETS_bios.pkl")
# except:
#     pass

df = pd.read_pickle("classification-data/FULL_BOTH_TWEETS.pkl")
df2 = pd.read_pickle("classification-data/FULL_LIFE_BIOS.pkl")

