# opinion-dissem
Description: Component of the Nonlinear and Complex System Research Group's 'Information Cascades' project focusing on assigning opinion values to tweets and users. All sensitive information has been removed from the new public version of this repository.

## Summary

This repository includes the code for converting Twitter followership networks into csv's and visual graphs.

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

Poster Link: (Brief Overview)
https://drive.google.com/file/d/1Mi2hnEhycsTwkPQ48KotmL8sXR4MsQhk/view?usp=sharing

Journal Link: (Too much information) https://drive.google.com/file/d/118xRUz1HUETt3eHmt-tr7rNlCHFY5uvD/view?usp=sharing


## Repository Guide

(TBD)
