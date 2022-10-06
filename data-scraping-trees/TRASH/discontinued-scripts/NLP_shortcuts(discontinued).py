"""A ton of helper functions and modules for NLP. Text cleaning, getting textblob values, etc."""

"""DISCONTINUED VERSION OF NLP_preprocessing.py"""
"""
____________________________________________
      TEXT CLEANING / PRE-PROCESSING
--------------------------------------------
"""
import re
import textblob as TextBlob

def clean_tweet(text):
    """Cleans tweets for tweet-specific terms"""
    # Tweet specific cleaning
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # get rid of tags ('r' tells python it's a raw string)
    text = re.sub(r'#', '', text) # removes the # symbol in front of hashtags
    text = re.sub(r'RT[\s]+', '', text) # removes 'RT' for retweets
    text = re.sub(r'https?:\/\/\S+', '', text) # Removes hyperlinks question mark --> can have 0 or 1 s's
    return text

def clean_text(text):
    """Cleans text."""

def subjectivity(text):
    """Returns (0,1) subjectivity score for tweet text. Subjectivity quantifies the amount of personal opinion 
    and factual information contained in the text. The higher subjectivity means that the text contains personal 
    opinion rather than factual information.  Polarity lies between [-1,1], -1 defines a negative sentiment and 
    1 defines a positive sentiment."""
    return TextBlob(text).sentiment.subjectivity

def polarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(text):
    return TextBlob(text)

def correctEmojis(text):
    pass