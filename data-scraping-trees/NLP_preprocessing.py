"""
Description:
This file contains lots of helper functions and modules for NLP, especially preprocessing (text cleaning, getting textblob values, etc.). 
I used the online Introduction to NLP course by WomenWhoCode on youtube and github: https://github.com/WomenWhoCode/WWCodeDataScience/tree/master/Intro_to_NLP.

(Author: Kerria Pang-Naylor)
"""

# IMPORTS
import pandas as pd
import re
from textblob import TextBlob
import contractions

from sklearn.utils import shuffle

import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

import string

from nltk.corpus import wordnet

#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
import ast


"""
____________________________________________
      TEXT CLEANING / PRE-PROCESSING
--------------------------------------------
"""

def spell_correct_alt(text_tokenized):
    """
    Some words of erroneously corrected, but this is later reversed in "last_clean".
        INPUTS: text_tokenized (list of words (strings))
        RETURNS: corrected (list of corrected words)
    """
    spell = SpellChecker()
    mispelled = spell.unknown(text_tokenized)

    correct_d = {word: spell.correction(word) for word in mispelled}

    corrected = [correct_d[word] if word in correct_d else word for word in text_tokenized]
    return corrected




#  NLTK's word lemmatizer requires POS tags in wordnet fromat
def get_wordnet_pos(tag):
    """Convert's POS tags to wordnet's format. Only covers some tags
            INPUTS: NLTK POS tag 
            RETURNS: Wordnet POS tag"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else: 
        return wordnet.NOUN


def remove_emojis(text):
    """
    Removes emojis from text. Taken from medium article.
        INPUTS: text (str)
        RETURNS: str 
    """
    emoji_pattern = re.compile("["
                           u"\U0001F300-\U0001FAF6" # emoticons \U0001F600-\U0001F64F
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_tweet(text):
    """Cleans tweets for tweet-specific terms. Inputs and outputs string."""
    # Tweet specific cleaning
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # get rid of tags ('r' tells python it's a raw string)
    text = re.sub(r'#', '', text) # removes the # symbol in front of hashtags
    text = re.sub(r'RT[\s]+', '', text) # removes 'RT' for retweets
    text = re.sub(r'https?:\/\/\S+', '', text) # Removes hyperlinks question mark --> can have 0 or 1 s's
    text = re.sub(r'\n', '', text)
    text = remove_emojis(text)
    
    return text

def preprocess_df(filename = "classification-data/FULL_BOTH_TWEETS_CLASSIFICATION", df = None, text_name = "text"):
    """
    Preprocessing for dataframe. Adds new columns to dataframe. Returns new dataframe. 
    text_name and category_name are names of columns in dataframe
        INPUTS: filename (str) or df (pandas dataframe), text_name (str), category_name (str)
        RETURNS: cleaned (pandas dataframe)
    """
    if df == None:
        df = pd.read_pickle(f"{filename}.pkl")

    
    df["cleaned_tweet"] = df[text_name].apply(lambda text: clean_tweet(text))
    # first remove contractions
    df["no_contract"] = df["cleaned_tweet"].apply(lambda text: [contractions.fix(word) for word in text.split()])

    # turn list of contractionless words back into a singular string
    df['msg_str'] = [' '.join(map(str,list_)) for list_ in df['no_contract']]
    
    # remove punctuation
    punc = string.punctuation.replace("-","") # punctuation without "-"
    df["no_punc"] = df["msg_str"].apply(lambda s: s.translate(str.maketrans('-', ' ', punc))) # remove all punc except '-' which is replaced with a space

    # test remove all but letters (REMOVED)
    df["correct_rest"] = df["no_punc"].apply(lambda text: " ".join(re.findall("[a-zA-Z,.]+",text)) ) # get rid of rest of weird characters
    
    # make new tokenized column
    df['tokenized'] = df['correct_rest'].apply(word_tokenize)
    df["manual_correct"] = df['tokenized'].apply(last_clean)

    # make everything lowercase
    df['lower'] = df['manual_correct'].apply(lambda list_: [word.lower() for word in list_])

    # get rid of punctuation (REPLACED)
    #punc = string.punctuation
    #df['no_punc'] = df['lower'].apply(lambda list_: [word for word in list_ if word not in punc])

    # spell checking (we're not doing this rn)
    # from spellchecker import SpellChecker # microsoft text blob
    # spell = SpellChecker()

    # get rid of stop words

    stop_words = set(stopwords.words('english'))
    df['stopwords_removed'] = df['lower'].apply(lambda list_: [word for word in list_ if word not in stop_words])
    
    # lemmatization
    df['pos_tags'] = df['stopwords_removed'].apply(nltk.tag.pos_tag)

    df['wordnet_pos'] = df['pos_tags'].apply(lambda list_:[(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in list_])

    wnl = WordNetLemmatizer()

    df['lemmatized'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])


    return df

def last_clean(tokenized_text):
    """
    Last minute removals and corrections of words. 
        INPUTS: tokenized_text (list of words)
        RETURNS: cleaned (list of words)
    """
    # cleaned = [word for word in ast.literal_eval(tokenized_text) if word != "amp"]
    # cleaned = [word if word != "scouts" else "scotus" for word in cleaned ]
    # cleaned = [word if word != "scouts" else "scotus" for word in cleaned ]
    cleaned = []
    tokenized = tokenized_text
    skip = {i : False for i in range(len(tokenized))} # booleon dictionary of whether certain indices should be skipped

    for i, word in enumerate(tokenized) :
        word = word.lower()
        if skip[i]:
            continue

        if (word != "amp") and (word != "scouts") and (word != "u") and (word != "proline") and (word != "pro") and (word != "roe") and (word != "roevswade") and (word != "rap") and word != "v":
            cleaned.append(word)

        elif (word == "amp") or (word == "u") or (word == "v"):
            pass

        elif word == "scouts":
            cleaned.append("scotus")

        elif word == "proline":
            cleaned.append("prolife")

        elif word == "pro":
            # make ['pro', 'life'] --> prolife (pro choice --> prochoice)
            try:
                if tokenized[i+1] == "life":
                    cleaned.append("prolife")
                    skip[i+1] = True
           
                elif tokenized[i+1] == "choice":
                    cleaned.append("prochoice")
                    skip[i+1] = True

                else:
                    cleaned.append(word)
            except Exception:
                pass

        elif word == "roe":
            # replace ['roe', 'v', 'wade'] to roevwade
            try:
                if ("v" in tokenized[i+1]) and (tokenized[i+2] == "wade"):
                    cleaned.append("roevwade")
                    skip[i+1], skip[i+2] = True, True
            except Exception: # when tries to go out of index, don't replace with "roevwade"
                pass
        elif word == "roevswade":
            cleaned.append("roevwade")
        elif word == "rap":
            cleaned.append("rape")

        

    return cleaned


def subjectivity(text):
    """
    Returns (0,1) subjectivity score for tweet text. Subjectivity quantifies the amount of personal opinion 
    and factual information contained in the text. The higher subjectivity means that the text contains personal 
    opinion rather than factual information.  Polarity lies between [-1,1], -1 defines a negative sentiment and 
    1 defines a positive sentiment.
        INPUTS: text (str)
        RETURNS: str
        """
    return TextBlob(text).sentiment.subjectivity

def polarity(text):
    """
    Returns (-1,1) polarity score for tweet text (-1 being very negative sentiment and +1 being very positive).
        INPUTS: text (str)
        OUTPUTS: (str)
    """
    return TextBlob(text).sentiment.polarity



