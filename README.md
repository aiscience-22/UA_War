 # War in Ukraine: Sentiment Prediction

**Using Artificial Intellect to discover Twitter trends and Sentiment Analysis**

We select to discover, how War in Ukraine is presented on Twitter, and to realize Sentiment Prediction for current twits. This subject was selected due to topicality of that information and acute social need for such an analysis.

On February 24th, 2022, Russia invaded Ukraine after months of military preparation around the borders. The war is going to have huge impacts around the world, perhaps even ending the globalized era as we know it. It is imperative that we capture and analyze the massive amounts of data being put out as a result of this war and extract insights for future generations.

### Hypothesis

We intend to show that Ukrainian War news have impact on the trend of tweets negative sentiment. Our supervised machine learning model might predict, based on tweet data and information about news, whether a tweets negative sentiment will rise or not.

For first segment we will use datetime and lenght of tweets for prediction, but later we could decide to make clasterisation for Twitter users for more accurate trend prediction.

### Description of the communication protocols

![repo_org.png](/Notes%20/repo_org.png) 


1. All premliminary data research and transformation live in the Preliminary Data Analysis - Directory
2. Main project files that consist of combining the various data set together and machine learning of data will live on the top level of the directory
3. All support data files and images live in a sub directory under the main Resources directory.

### The team roles for first segment:  
**Olga Podolska** - mockup datasets, twitter data, RoBERTa sentiment analysis, machine leanning  
**Veronica Lobkina** - data base 
**Jaymee Liu** - GitHub 
**Jesse Hernandez** - data base, machine learning


### Overview of the News Data Analysis

For first segment we created mockup dataset with main news just to use it as a placeholder for future web scraping or API downloading. 


## Overview of the Twitter Data Analysis

The source of data is [Ukraine Conflict Twitter Dataset ](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows?select=0801_UkraineCombinedTweetsDeduped.csv.gzip) with 49.74M tweets on Kaggle, 12 Gb. 

![img1.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img1.png) 

We decided to limit ourselves to dates from February 24 (starting war date) to July 12 (starting project date).
We hoped to answer the following question:
* When were the earliest and latest tweets in this dataset created?  
* Visualize tweet frequency by date  
* How many languages are in this dataset?  
* What percentage of the tweets is in English (en)?
* How many symbols the shortest and longest tweet contains? 
* What a location of Twitter users in this Dataset?
* What the score for each of negative, neutral, and positive sentiments?
* Which model we should use first for prediction? 
* How we can improve prediction?

# Data exploration phase of the project

## Twitter data preprocessing

Totally we have 161 files with the twits about Russian-Ukrainian War, totally 12 Gb, but it is too much data for the first segment our project, where we want just play with the data and draft out project. Due to that we choose preprocess for start the August data only and see what can we say about them. 

The raw dataset for 12 days of August has 1849926 rows and 28 columns. The earliest tweet was at 2022-08-01 00:00:00, and the latest was at 2022-08-12 23:59:58, which means that data is correct, it is all August tweets. 

We can see, that amount of tweets and tweets frequency is significantly different day by day. Probably this difference correlate with the state of war, events and news about it, but this hypothesis needs to be tested. 

![img4.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img4.png) 

About used languages we can see that English (en) was the far the most prevalent language in this dataset, nearing 1.2 million tweets out of 1.96 million. The second and third most prevalent languages were French and Thai, respectively.

Note that the forth most prevalent language was "und", which is used to indicate that Twitter could not detect a language. We can assume, that it safely inspect English language only due to most of tweets are in English despite they are from different corners of world: USA, UK, India, Ukraine etc.

![img5.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img5.png) 

Shortest tweet has 1 character. Longest tweet has 1027 characers, despite a tweet can have 280 characters max. How could one have more than the limit? Upon research, mentions supposedly do not count toward the character limit when the tweet is a reply. The distribution of lengths is right-skewed. Most tweets appear to be below 300 characters in length. But because we have a few outlying tweets that have anomalously long lengths, as investigated above, the histogram has an elongated x-axis

![img6.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img6.png) 

Dataset has 1849926 rows, but column coordinates has 1847412 null values, therefore we have 2514 data here, which is basically nothing. We can safely remove this column as well.

Column location has 800249 null values, it means that we have data in more then million rows. There are 127675 unique locations in this DataFrame.As we can see, the location input wasn't formalized and even when users filled it, they fill it with some creative description of location as "Facing West" or "The Peanut Gallery", which don't give geographic information for us. Therefore, despite analyzing correlation between twit's sentiment and autor's geographic location would be great idea, unfortunately we cannot realize it. Location data wasn't standardized and only can give us information about users' endless fantasy.

![img2.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img2.png) 

![img3.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img3.png) 

## RoBERTa Sentiment Analysis

For setiment analysis we tried to use August tweets dataset containing 1050085 rows and 2 columns (datetime and text only). For each tweet the pretrained RoBERTa model planed to generate a score for each of negative, neutral, and positive sentiments.

Unfortunately, the predicted time to complete this task turned out to be 59 hours,and we were forced to stop RoBERTa after 6 hours of work.

![img11.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img11.png) 

However, we have received enough data to our mockup twitter dataset:

![img12.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img12.png) 

## Redusing data

After some research we discovered: there are two popular free Twitter datasets only. Both of them are about 12-13 Gigabytes. All smaller datasets are broken in different ways, I believe I tried all of them in the whole Internet. We can just assume that it is normal amount of data for this Ukrainian war related tweets, and there is no point for us do our own API downloading, it will be the same big.

How can we reduce data?

As example, we can use tweets with only one hashtag. Unfortunately, it definitely will skew our data, hashtags popularity changes all the time. We don’t even sure a certain hashtag was existing at Feb 24! (Most probably it don't.)

The same problem is with many another ways of reducing. We don’t know users gender, age, geographical position to make a selection according to these parameters. Any selection according to the content of the tweet - quote, retweet, hashtags – skew out data.


But we know for sure one thing about users: data of their registration. We can take tweets from the registered before certain data users only. 

First, it allow as meet out technical requirements.
Second, we can be sure that there is no fake Kremlin users in this dataset: now there is Kremlin's network of Twitter accounts that work together to retweet and drive up traffic:  

[Kremlin's Network](https://www.bbc.com/news/technology-60790821)

![img7.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img7.png)

Additionally, we can compare our study for reduced dataset with Alexander Shevtsov study for the whole dataset. If significant difference in trends  appears, that means we catched the Kremlin bots working!  

[Alexander Shevtsov study](https://alexdrk14.github.io/RussiaUkraineWar/sentiment.html)  

![img8.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img8.png)

We decided to choose 2008 year of registration as a limit. The duration of the war is 179 days for today. Our training dataset is 12 days, i.e. 7%.That means we should be prepared to work with 14 times bigger dataset. Dataset with the tweets from users registered before 2008 has 3266 rows.

Additionally, 2008 is a year when Russian invasion to Georgia started, and probably Kremlin network starts to develop.

## Joining the data

For our research we needed to join the twitter table with the news table. We used PostgreSQL for that:

![img13.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img13.png)

The result is the table with the date, tweets, sentiment of tweets and weight of events:

![img14.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img14.png)

## Description of feature engineering and the feature selection

Our team considered different options: Supervised and Unsupervised learning, Clastering, different Neural Networks for sentiment analysis. 

For sentiment analysis we eventually chose RoBERTa, because it is a free pretrained model from Meta (Facebook), and it is robustly optimized method for pretraining natural language processing (NLP) systems that improves on Bidirectional Encoder Representations from Transformers, or BERT, the self-supervised method released by Google in 2018. 

For discovering trends and prediction we chose to use Supervised Machone Learning, considering that as result of prediction we will have negative sentiment, which is a number. We used Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier to compare the results. 

As we intended to show that Ukrainian War news have impact on the trend of tweets negative sentiment, we need to predict the number of this sentiment for the days in our prediction dataset. Our supervised machine learning model might predict, based on tweet data and information about news, whether a tweets negative sentiment will rise or not. So far we have managed the machine learning model to return us to labels:

For first segment we used datetime and lenght of tweets for prediction, but later we could decide to make clasterisation for Twitter users for more accurate trend prediction.

  

## Technologies, languages, tools, and algorithms used throughout the project:

**Used tools:**
numpy                     1.21.5  
pandas                    1.16.0  
PostgreSQL  
pgAdmin                   4  
scipy                     1.7.3  
scikit-learn              2.2.0  
imbalanced-learn          0.9.0  
RandomOverSampler  
SMOTE algorithms  
ClusterCentroids algorithm  
SMOTEENN algorithm  
BalancedRandomForestClassifier (bias reduction model)  
EasyEnsembleClassifier (bias reduction model)   
transformers  
RoBERTa (pretrained deep neuron network model)  