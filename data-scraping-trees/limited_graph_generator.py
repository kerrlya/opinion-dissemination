"""
Description:
Script that creates followership graphs and tweet cascades while only considering a limited number of users. This creates
interconnected graphs in a shorter time span (since scraping takes a shorter time).

(Author: Kerria Pang-Naylor)
"""

import networkx as nx
from twitter_user_class import *
import ast
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot

def followerDF_to_graph(filename,df = None):
    """
    Takes in filename of followerDF csv., exports graph showing followerships. May accept pre-generated graph with only nodes
        INPUTS: filename (string, no .csv, name of graph .gexf file you want to expot), 
                df (optional, pandas datafram)   
        RETURNS: None (shows graph, only writes gexf file with filename.gexf) 
    """
    if not df:
        df = pd.read_csv(f"{filename}.csv")
    followerG = nx.DiGraph()

    for index, row in df.iterrows():
        user1, user2 = ast.literal_eval(row["Pairs"])

        if row["Follows"]:
            followerG.add_edge(user2, user1)
        
        if row["Followed_by"]:
            followerG.add_edge(user1, user2)
    nx.write_gexf(followerG, "100node_HMC_network.gexf")
    nx.draw(followerG, with_labels = True)
    plt.show()



def gen_cascades(tweets_filename = "HMC_large_cascade_data", followership_filename = "valid_HMC_followershipP", limit_L = None):
    """
    Generates cascade of tweets from tweets_filename.csv based on timestamp and followership information. 
    The dataframes must have a column of "epoch_time" and "usernames".
        INPUTS: tweets_filename (string, no .csv) - used for tweet IDs and epoch_time to generate cascade 
                followership_filename (string, no .csv) - used to determine followership relationship to 
                    see if two tweets may be connected
        OUTPUTS: g (networkx graph datatype)
    """
    # limit_L currently not being used
    tweet_df = pd.read_csv(f"{tweets_filename}.csv")
    followee_df = pd.read_csv(f"{followership_filename}.csv")
    df = tweet_df.sort_values(by="epoch_time", ascending = False)

    g = nx.DiGraph()

    g.add_nodes_from(list(tweet_df["usernames"]))

    for i in range(len(df)-2):
        df_temp = df[i:]
        root_user = list(df_temp["usernames"])[0]
        #print(f"\n{root_user} ")
            # if "RT" in row["tweet_text"]:
            #     root_user = user(tweetID = row["conversation_id"]).username
            #     g.add_edge(root_user, user_, color = "blue")
        for index, row in df[1:].iterrows():
            other = str(row["usernames"])
         
            # if friends_with(other, root_user)[0]:
            #     g.add_edge(root_user, other)
            try:
                if is_following(followee_df, other, root_user): # since we are giving a followee_df, this actually gives us if other is being followed by the root_user
                    g.add_edge(other,root_user)
            except Exception:
                print(f"Exception with {root_user} and {other}")

    nx.draw(g, with_labels = True)
    plt.show()

    return g

def helper_fol_2(following_list_str, include_L):
    """
    Helper for followeeship_DF_to_graph. Returns list of those following who are in the set of include_L
        INPUTS: following_list_str (string of a list of usernamesfrom csv)
                include_L (list of usernames you want to include )
        RETURNS: L_rel (list)
    """
    if following_list_str != "invalid user":
        L = ast.literal_eval(following_list_str)
        L_rel = [user_ for user_ in L if (user_ in include_L)]
        return L_rel
    else:
        return "invalid"


def followeeship_DF_to_graph(filename, include_L, export = True):
    """
    Takes different form of followership df (one with full lists of who each node in following), and converts to a gexf graph.
    Try filename = 'graph-datasets/HMC_following_followership.csv'
        INPUTS: filename (str, no .csv), include_L (list of usernames you should include), export (bool, if True, export graph to .gexf)
        RETURNS: g (networkx graph of followership )
    """
    df = pd.read_csv(f"{filename}.csv")
    g = nx.DiGraph()

    for index, row in df.iterrows():
        user_ = row["user"]
        print(user_)
        L_rel = helper_fol_2(row["following_list"], include_L) # list of relevant followees 

        if L_rel != "invalid":
            for followee in L_rel:
                g.add_edge(user_, followee)
    
    nx.draw(g, with_labels = True)
    plt.show()

    if export:
        nx.write_gexf(g, f"{filename}.gexf")

    return g


# filename = "test_followership_HMC"

# G = nx.read_gexf("100HMCNetwork.gexf")
