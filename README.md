# opinion-dissemination
Description: Component of the Nonlinear and Complex System Research Group's 'Information Cascades' project focusing on opinion mining (i.e., assigning opinion values to tweets and users). All sensitive information has been removed from the new version of this repository.

## Summary

This repository includes the code for converting Twitter followership networks into csv's and visual graphs.

(Here is a 350 node graph generated from the relationships among all the followers of @HMC_Alumni)

<img width="500" alt="Screen Shot 2022-09-14 at 7 08 55 PM" src="https://user-images.githubusercontent.com/58631280/194263739-f29d1bbf-37c3-496e-a684-c07f009d4c87.png">

I also have an experimental script in tweet_scraper.py that extracts and labels large (>5000 point) labeled datasets of tweets by inputting any number popular twitter accounts whose followers' tweets correspond to a certain keyword.

This summer, I created labeled datasets twenty-thousand tweets and ten-thousand Twitter bios. I first pre-labeled two sets of users who followed either @NARAL or @March_for_life as pro-choice or pro-life (respectively). We then extracted and correspondingly labeled any tweets from these users within 10 days after the overturn that contained the keywords: "roe", "wade", "abortion", or "unborn".

The 'shortcut' here is that the program assumes that all followers of a Twitter user hold a certain belief. So far, this hasn't created any extensive issues. The classification models run with 80-86% accuracy and the language analysis visualizations make sense with what you would expect.

![uncleaned_balanced-dup_BIO](https://user-images.githubusercontent.com/58631280/194277401-56a2001c-82b8-411c-adc1-ca0eafe4f7b0.png)

I also built some custom too that use NLTK and textblob to perform language analysis and visualization. Currently, we have analyses on the Pro-Choice and Pro-Life tweets/user bios from Twitter's reaction to the 2022 Overturn of Roe v. Wade. (Though these can be applied to any dataset.)

Top words in Pro-Life bios:
<img width="1279" alt="20_top_nonstopwords_LIFE" src="https://user-images.githubusercontent.com/58631280/194264897-aee1eac5-b5e8-419d-8ec8-f4f276356403.png">

Top words in Pro-Life bios:
<img width="1273" alt="20_top_nonstopwords_CHOICE" src="https://user-images.githubusercontent.com/58631280/194265748-40838a23-c2f0-402f-86b4-0ea8410f027f.png">

## More Information
Detailed documentation can be found with the following links.

[Summer Poster Link](https://drive.google.com/file/d/1Mi2hnEhycsTwkPQ48KotmL8sXR4MsQhk/view?usp=sharing) (Brief Overview)

[Summer Journal Link](https://drive.google.com/file/d/118xRUz1HUETt3eHmt-tr7rNlCHFY5uvD/view?usp=sharing) (Too much information) 

[Fall Research Journal Link]()

## Repository Guide
This is a directory guide to the data-scraping-trees folder.

This folder contains scripts, datasets, plots, and graphs relating to data scraping from Twitter and data-based graph generation.

----------------------------------
     SUB-FOLDER DESCRIPTIONS
----------------------------------
(look in code mode for this)
__pycache__ 
        DESCRIPTION: Automatically generated caches for all tweepy scraping tools/scripts in data-scraping-trees
better_perf_models
        DESCRIPTION: contains all bio and tweet classification models with varying number of features
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
classification_perf_data
        DESCRIPTION: Performance data and graphs of testing different models
    graphs
                DESCRIPTION: contains scatterplots of models' number of features vs. accuracy and number of features vs. cross-entropy
    test_dataframes
                DESCRIPTION: contains respective dataframes of scatterplots in 'graphs' folder
    
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
external-datasets
        DESCRIPTION: contains outside datasets from the internet
graph-datasets
        DESCRIPTION: contains datasets of followership and twitter cascade networks 
            _followership_combinations - dataset that contains every single combination pair of connected users and their followership relationship

graphs
        DESCRIPTION: pictures and gexf files of twitter cascade and followership networks
TRASH
        DESCRIPTION: discontinued scripts and datasets (many of these lack documentation)
