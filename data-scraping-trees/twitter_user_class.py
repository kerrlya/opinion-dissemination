"""
Description:
This file contains the user class, which provides shortcuts to scrape information about a certain twitter 
user from their twitter ID, username, or tweet. It also contains a number of useful functions for determining
followership relationships. This script uses both v1 and v2 of the twitter API

(Author: Kerria Pang-Naylor)
"""
import nltk
import random
import tweepy
from textblob import TextBlob
import config_tweepy
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import time
from itertools import *
import datetime as dt
import math
import ast


client = tweepy.Client(bearer_token = config_tweepy.BEARER_TOKEN, wait_on_rate_limit=True) # initialize tweepy client

auth = tweepy.OAuthHandler(config_tweepy.CONSUMER_KEY, config_tweepy.CONSUMER_SECRET) # setup api v1
auth.set_access_token( config_tweepy.ACCESS_TOKEN,  config_tweepy.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True)

class user:
    """Class system with information about users"""

    def __init__(self, username = None, id = None, tweetID = None,user_object = None):
        """Initialize user object. Can initialize with either username, twitter id (id), the ID of a tweet the user has written (tweetID), or a pre-existing user object"""
        
        user_fields = ["created_at","description","entities","location","name","pinned_tweet_id", "protected","public_metrics","url","username","verified","withheld"]

        #first get the twitter user object with max amount of user_fields
        if id != None:
            self.t_object =  client.get_user(id = id, user_fields = user_fields) #twitter object of user
        elif username != None:
            self.t_object =  client.get_user(username = username, user_fields = user_fields) #twitter object of user
        elif tweetID != None:
            tweet = client.get_tweet(id = tweetID, expansions = ["author_id"], user_fields = user_fields)
            temp_id = tweet.data.author_id
            self.t_object = client.get_user(id = temp_id, user_fields = user_fields)
        elif user_object != None:
            self.t_object = user_object

        self.id = self.t_object.data.id # twitter user ID
        self.username = self.t_object.data.username # twitter username
        
        self.bio = self.t_object.data.description # twitter bio
        self.location = self.t_object.data.location # location/city (if disclosed)
        self.entities = self.t_object.data.entities # twitter entities (twitter analyzed subjects of the user)
        self.name = self.t_object.data.name # name of user

        public_metrics = self.t_object.data.public_metrics
        self.num_followers = public_metrics["followers_count"]
        self.num_following = public_metrics["following_count"]
        self.num_tweets = public_metrics["tweet_count"]
        self.num_listed = public_metrics["listed_count"]

        self.protected = self.t_object.data.protected

        try:
            self.pinned_tweet = client.get_tweet(id = self.t_object.data.pinned_tweet_id) # pinned tweet (if posted)
        except:
            pass
        
        self.is_verified = self.t_object.data.verified # whether or not user is a verified user or not
        self.relationships = {} #  API v1 method of listing the relationship (only useful for small sets of people)

    def is_valid_user(self):
        """Returns booleon showing that the user meets/doesn't meet certain parameters with following and follower count to be 'valid' """
        if (self.num_followers > 5) and (self.num_following > 5) and (self.num_tweets >= 0) and not self.protected:
            return True
        else:
            return False

    def getFollowersList(self, amount = "ALL", searchFor = None):
        """(BETTER TO USE NON-CLASS VERSION) Class function. Returns list of usernames the user is following. May also use to see if self is 
        following username in 'searchFor'."""
    
        if amount == "ALL": 
            num_max = 130 # max of 130*1000 users
        else:
            num_max = math.ceil(amount/1000)

        pages = [] # list of returned pages for all followers
        followers_usernames = [] # list of follower usernames
        for response in tweepy.Paginator(client.get_users_followers, self.id, max_results=1000, limit= num_max):
            pages.append(response)
            time.sleep(random.randint(2,6))
  
        for page in pages:
            for user_ in page[0]:
                followers_usernames.append(user_.username)

        # start search section
                if str(user_.username) == searchFor:
                    #print("looked through ", len(followers_usernames), " followers")
                    return True
                
            if searchFor != None:
                return False


        # end section

        self.followers = followers_usernames
        return followers_usernames
    
    def isFollowing(self, other_username):
        """DISCONTINUED (use friends_with for full relationships between two users). Returns booleon of whether self is following inputted 'other_username' 
            (str, twitter usename) """
        other = user(username = other_username)
        
        try:
            if len(self.following) == self.num_following:
                if other_username in self.following:
                    
                    return True
                else:
                    return False
    
        except:

            if self.num_following <= other.num_followers:
                return self.getFollowingList(searchFor = other_username) # will return True if search for is in the list of ppl you're following, or a list of all the pp

            else:
                return other.getFollowersList(searchFor = self.username)
        
    def isFollowed(self, other_username):
        """DISCONTINUED (use friends_with for full relationships between two users). Returns true if other_username is following self."""
        other = user(username = other_username)

        return other.isFollowing(self.username)

    def demOrRep(self):
        """ DISCONTINUED (this method is very slow and impractical for large datasets)
        Determines if self is more democrat or republican-leaning based on the ratio between republican and democratic candidates they are following.
        Returns a scalar between -1 and 1 (-1 is democratic, +1 is republican). """
        self.getFollowingList()

        self.num_dem_following = 0
        self.num_rep_following = 0 

        for dem in user.dems:
            if dem in self.following:
                self.num_dem_following += 1
        
        for rep in user.reps:
            if rep in self.following:
                self.num_rep_following += 1
        
        
        
        if self.num_dem_following > self.num_rep_following:
            self.pol_score = -self.num_dem_following/(self.num_dem_following + self.num_rep_following)
            self.isDem = True
            self.isRep = False
            

        elif self.num_dem_following < self.num_rep_following:
            self.pol_score = self.num_rep_following/(self.num_dem_following + self.num_rep_following)
            self.isRep = True
            self.isDem = False
            
        elif (self.num_dem_following ==0) and (self.num_rep_following == 0):
            self.pol_score = "apolitical"
            
        
        else:
            self.pol_score = 0
        return self.pol_score    

    def friends_with(self, other_username):
        """API v1 of determining the full relationship between self and other_username. """
        friend_object = api.get_friendship(source_screen_name = self.username, target_screen_name = other_username)  # returns twitter api tuple
        self.relationships[other_username]  = {"follows": friend_object[0].following, "followed by": friend_object[0].followed_by, "api tuple": friend_object} #store this for good measure (?)
    
    
    def is_following(self, other_username):
        """Returns True if self is following other_username, false otherwise. Doing this will store the relationship in"""
        if other_username in self.relationships:
            return self.relationships[other_username]["follows"]

        self.friends_with(other_username)
        return self.relationships[other_username]["follows"]

    def followed_by(self, other_username):
        """Returns True if self is followed by other_username, false otherwise."""
        if other_username in self.relationships: # if already in dictionary, just return its value
            return self.relationships[other_username]["followed by"]
        
        self.friends_with(other_username) # otherwise, make the dictionary entry
        return self.relationships[other_username]["followed by"]
    
    def dem_or_rep(self):
        """DISCONTINUED. V1 version of getting score. doesn't work :( issue with finding friendships with certain usernames """  
        if self.protected:
            return "protected account"
        self.num_dem_following = 0
        self.num_rep_following = 0 

        for dem in user.dems:
            try:
                if self.is_following(dem):
                    self.num_dem_following += 1
            except:
                print("Issue with ", dem)

        
        for rep in user.reps:
            try:
                if self.is_following(rep):
                    print(rep)
                    self.num_rep_following += 1
            except:
                print("Issue with ", rep)


        if self.num_dem_following > self.num_rep_following:
            self.pol_score = 1
            return 1

        elif self.num_dem_following < self.num_rep_following:
            self.pol_score = -1
            return -1
        elif (self.num_dem_following ==0) and (self.num_rep_following == 0):
            self.pol_score = "apolitical"
            return "apolitical"
        
        else:
            self.pol_score = 0
            return 0 


def friends_with(username, other_username, relationships = None):
    """v1 helper for new is_followed and is_following information stored in dictionary"""
    try:
        friend_object = api.get_friendship(source_screen_name = username, target_screen_name = other_username)  # returns twitter api tuple
        #dict_out = {f"{username} follows {other_username}": friend_object[0].following, f"{username} followed by {other_username}": friend_object[0].followed_by, f"{username}, {other_username} api tuple": friend_object}
        #time.sleep(random.randint(2,5))

        if isinstance(relationships,dict):
            #relationships[f"{username}; {other_username}"] = [friend_object[0].following, friend_object[0].followed_by] # [if username followers other, if username is followed by other]
            #relationships["users"].append([f"{username}", f"{other_username}"])
            relationships["following"].append(friend_object[0].following)
            relationships["followed by"].append(friend_object[0].followed_by)
            # I changed the design so this if statment doesn't ever get used^^^

            return relationships
 
        else:
            # using this instead vvv
            #print("no dict")
            return  (friend_object[0].following, friend_object[0].followed_by) #"api tuple": friend_object}
    
    except Exception:
        print("V1 doesn't work :( (or protected user)")
        if relationships:
            return relationships
        else:
            return (np.nan, np.nan)

def get_followership_df(user_L, filename = None):
    """
    Returns Followership dataframe of ALL POSSIBLE PAIR COMBINATIONS based on list of users. This method is ideal for
    smaller sets of users where the number of pair combinations is relatively small.
        INPUTS: user_L (list of usernames), filename (str, name of dataframe you will export)
        RETURNS: df (pandas dataframe)
    """

# BELOW COMMENTED OUT WAS THE "FAST" WAY THAT DOESN'T LET YOU SEE THE DF UNTIL ITS FULLY COMPLETED. I'LL REIMPLEMENT THIS WHEN I HAVE THE RATE LIMIT FIGURED OUT
    # df = pd.DataFrame({"Pairs" : [], "Follows" : [], "Followed_by" : [], "Relationship": []})
    # df["Pairs"] = get_unique_combinations(user_L)
    # df["Relationship"] = df["Pairs"].apply(lambda pair: friends_with(pair[0], pair[1]) )
    # df["Follows"] = df["Relationship"].apply(lambda x: x[0])
    # df["Followed_by"] = df["Relationship"].apply(lambda x: x[1])

    #follows, followed_by = friends_with(pair[0], pair[1])
    #df.assign(Follows = )

# SLOW WAY
    df = pd.DataFrame({"Pairs" : [], "Follows" : [], "Followed_by" : []})
    df["Pairs"] = get_unique_combinations(user_L)

    for index, row in df.iterrows():
        user1, user2 = row["Pairs"]
        follows, followed_by = friends_with(user1, user2)

        df.loc[index,"Follows"] = follows
        df.loc[index,"Followed_by"] = followed_by

        if filename:
            df.to_pickle(f"{filename}.pkl")

        time.sleep(random.randint(2,6))

    return df

def get_unique_combinations(user_L):
    """Helper to get_followership_df. Returns list of unique combination tuple pairs of users within list"""
    return list(combinations(user_L, 2))


def created_at_epoch(filename):
    """Creates new column with epoch timestamps for tweets.
            INPUT: filename (str, file name/path of pickle datframe)
            RETURNS: df (pandas dataframe)"""
    df = pd.read_csv(f"{filename}.pkl")
    df["epoch_time"] = df["created_at"].apply(to_epoch)
    df.to_pickle(f"{filename}.pkl")
    return df

def to_epoch(time_str):
    """Helper to created_at_epoch"""
    year, month, day, hour, minute, second = time_str[0:4], time_str[5:7], time_str[8:10], time_str[11:13], time_str[14:16], time_str[17:19]
    d = dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)).timestamp()
    return d

def getFollowingList(id = None, amount = "ALL"): 
    """Returns list of usernames the user is following. Non-class version. Use this construct large followership dfs. 
            INPUTS: id (int, twitter ID of user), amount (str or int, amount of followers)
            RETURNS: following_usernames (list of usernames that are following the user with Twitter ID - id) """

    if amount == "ALL": 
        num_max = 50
    else:
        num_max = math.ceil(amount/1000)

    pages = [] # list of returned pages for all followers
    following_usernames = [] # list of follower usernames
    for response in tweepy.Paginator(client.get_users_following, id, max_results=1000, limit=num_max):
                pages.append(response)
    
    for page in pages:
        for user_ in page[0]:
            following_usernames.append(user_.username)

    return following_usernames

def is_following(df, user1, user2, is_pickle = False):
    """Returns whether user1 is following user2 based on followership_df or df (full df with full list of followers) with "following_list" column.
            INPUTS: df (pandas dataframe), user1 (str, username of user1), user2 (str, username of user2), is_pickle (bool, determines whether or not each list
                    of followers will be expected to a be a string of a list or a list.
            RETURNS: bool (True if user1"""
    row = df.loc[df["user"] == user2]

    if not is_pickle:
        followers = ast.literal_eval(list(row["following_list"])[0])
    else:
        followers = list(row["following_list"])        
    if user1 in followers:
        return True
    else:
        return False



"""
-----------------------------------------
        USE EXAMPLES
-----------------------------------------
"""

# SCRAPING ALL FOLLOWERS OF 
# prolife = user(username = "March_for_Life")
# life_followers = prolife.getFollowersList()
# prochoice = user(username = "NARAL")
# choice_followers = prochoice.getFollowersList()


# df1 = pd.DataFrame({"NARAL" : choice_followers})
# df2 = pd.DataFrame({"March_for_life" : life_followers})
# df1.to_pickle("RVW_NARAL.pkl")
# df2.to_pickle("RVW_March_for_Life.pkl")


"""
-----------------------------------------
        DISCONTINUED SNIPPETS
-----------------------------------------
"""
# OLD WAY OF DETERMINING USER OPINIONS --> SEE WHETHER USER IS FOLLOWING MORE DEMOCRATS OR REPLICANS (ended up being too slow)
# dems = list(pd.read_excel("democrats.xlsx")["Username"]) # lists of democrats and republicans
# reps = list(pd.read_excel("republicans.xlsx")["Username"])

# add columns with the number of followers and following
# df = pd.read_pickle("classification-data/FULL_BOTH_TWEETS.pkl")

def get_followership_info(user_id):
    sleep_time = random.randint(1,3)
    time.sleep(sleep_time)
    
    try:
        user_obj = user(id = user_id)
        return {"num_followers" : user_obj.num_followers, "num_following": user_obj.num_following, "num_tweets" : user_obj.num_tweets}
    except Exception:
        return np.nan
# df["public_metrics"] = df["author_id"].apply(get_info)

# df.to_pickle("classification-data/FULL_BOTH_TWEETS_FOLLOWINFO.pkl")
# print("all done!")