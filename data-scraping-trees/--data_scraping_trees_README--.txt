This is a directory guide to the data-scraping-trees folder.

This folder contains scripts, datasets, plots, and graphs relating to data scraping from Twitter and data-based graph generation.

----------------------------------
     SUB-FOLDER DESCRIPTIONS
----------------------------------

__pycache__ 
        DESCRIPTION: Automatically generated caches for all tweepy scraping tools/scripts in data-scraping-trees

classification-data
        DESCRIPTION: contains the datasets used to train classification models
    BIOS-datasets 
            DESCRIPTION: contains training and preprocessing data for classification of sorting the bios of twitter users as pro-life and pro-choice
            preprocessing data includes the text cleaning process whereas the classification data only contains the raw text, label, and cleaned version of the text
    classification-models
            DESCRIPTION: contains joblib files for saved logistic regression classification of twitter bios and tweets (based on newer,
            larger 20000 tweet dataset, and the bios dataset extracted from it). These two joblib files are based on the uncleaned, balanced by duplication versions 
            of the models for both bio and tweet classification, as these had the highest accuracy rating.
    OLD-TWEETS-datasets
            DESCRIPTION: contains training and preprocessing versions of the older, smaller 10000 tweet dataset
    TWEETS-datasets
            DESCRIPTION: contains training and preprocessing versions of the newer, larger 20000 tweet dataset

confusion-matrices 
        DESCRIPTION: contains confusion matrices for different versions of the classifiers for bios and tweet classification
    bios
            DESCRIPTION: confusion matrices for bios classifiers
    tweets
            DESCRIPTION: confusion matrices for tweet classifiers
    old_classification_testing_data.txt
            DESCRIPTION: contains text confusion matrices for confusion matrices and accuracy rates based on  

EDA-data-plots
        DESCRIPTION: contains datasets and plots for/of exploratory data anlaysis for tweets and bios for the 20000 point and 10000 point datasets

graph-datasets
        DESCRIPTION: contains datasets of followership and twitter cascade networks 
            _followership_combinations - dataset that contains every single combination pair of connected users and their followership relationship

graphs
        DESCRIPTION: pictures and gexf files of twitter cascade and followership networks
TRASH
        DESCRIPTION: discontinued scripts and datasets (many of these lack documentation)