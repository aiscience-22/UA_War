 # War in Ukraine: Sentiment Prediction

**Using Artificial Intellect to discover Twitter trends and Sentiment Analysis**

We selected as our topic to discover how the war in Ukraine is presented on Twitter, and to run Sentiment Prediction for recent tweets. This subject was selected due to the urgency of the war and an acute social need for such an analysis.

On February 24th, 2022, Russia invaded Ukraine after months of military preparation around the border. The war is impacting the world in numerous ways, perhaps even ending the globalized era as we know it. It is imperative that we capture and analyze the massive amounts of data being put out as a result of this war and extract insights for future generations.

### Hypothesis

We intend to show that Ukrainian War news have an impact on the trend of tweets with negative sentiment. Our supervised machine learning model might predict, based on tweet data and information about news, whether tweets with negative sentiment will increase or not.

For the first segment we used simple linear regression, but at the next segment we could try to make clusterization for Twitter users for more accurate trend prediction.

### Description of the communication protocols

We created detailed [communication protocol](https://github.com/aiscience-22/UA_War/blob/main/Notes%20/Communication_protocol.md), which describes our ways of communication and team roles for the first two segments of project.

![repo_org.png](/Notes/repo_org.png) 

We created [Team Git Best Practices and Repo Organization](https://github.com/aiscience-22/UA_War/blob/main/Notes20/TeamGITBestPractices.md):
1. All preliminary data research and transformation live in the Preliminary Data Analysis - Directory
2. Main project files that consist of combining the various data set together and machine learning of data will live on the top level of the directory
3. All support data files and images live in a sub directory under the main Resources directory.

### The team roles for the first segment:  
**Olga Podolska** - mockup datasets, twitter data exploration and cleaning, RoBERTa sentiment analysis, machine learning model creating  
**Veronica Lobkina** - data base creating   
**Jaymee Liu** - GitHub creating  
**Jesse Hernandez** - data base interacting, GitHub conflicts resolving, machine learning model creating  

### The planned team roles for the second segment:  
**Olga Podolska** - RoBERTa sentiment analysis, machine learning improving  
**Veronica Lobkina** - data base integration  
**Jaymee Liu** - GitHub, twitter data exploration and cleaning, collecting data for events of war   
**Jesse Hernandez** - server installation, machine learning improvement 


### Overview of the News Data Analysis

For first segment we created a mockup dataset with main news just to use it as a placeholder for future web scraping or API download. 

## Overview of the Twitter Data Analysis

The source of data is [Ukraine Conflict Twitter Dataset ](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows?select=0801_UkraineCombinedTweetsDeduped.csv.gzip) with 49.74M tweets on Kaggle, 12 Gb. 

![img1.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img1.png) 

We decided to limit ourselves to dates from February 24th (starting war date) to August 12th.
We hoped to answer the following questions:
* When were the earliest and latest tweets in this dataset created?  
* Visualize tweet frequency by date  
* How many languages are in this dataset?  
* What percentage of the tweets are in English (en)?
* How many symbols do the shortest and longest tweet contain? 
* What are the locations of Twitter users in this Dataset?
* What are the scores for negative, neutral, and positive sentiments?
* Which model should we use for prediction? 
* How can we improve prediction?

# Data exploration phase of the project

## Twitter data preprocessing

In total, we have 161 files with the tweets about Russian-Ukrainian War, totaling 12 Gb, but it is too much data for the first segment of our project, where we want to just play with the data and draft out the project. Due to that, we chose to preprocess the August data only and see what we can glean from it. 

The raw dataset for 12 days of August has 1849926 rows and 28 columns. The earliest tweet was at 2022-08-01 00:00:00, and the latest was at 2022-08-12 23:59:58, which means that data is correct, the tweets are all from August. 

We can see, that the amount of tweets and their frequency is significantly different day by day. Most likely, this difference correlates with the state of the war, news about it, but this hypothesis needs to be tested. 

![img4.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img4.png) 

Regarding the language of tweets, we can see that English (en) was by far the most prevalent language in this dataset, nearing 1.2 million tweets out of 1.96 million. The second and third most prevalent languages were French and Thai, respectively.

Note that the forth most prevalent language was "und", which is used to indicate that Twitter could not detect a language. We can assume, that it safely inspects English language only due to most of tweets being in English despite coming from different corners of world: USA, UK, India, Ukraine etc.

![img5.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img5.png) 

The shortest tweet has 1 character. The longest tweet has 1027 characers, despite the fact that a tweet can have 280 characters max. How could this character limit be surpassed? Upon research, mentions supposedly do not count toward the character limit when the tweet is a reply. The distribution of lengths is right-skewed. Most tweets appear to be below 300 characters in length. Although, because we have a few outlying tweets that have abnormally long lengths, as investigated above, the histogram has an elongated x-axis.

![img6.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img6.png) 

The dataset has 1849926 rows, but column coordinates have 1847412 null values, therefore we have 2514 data points here, which is basically nothing. We can safely remove this column as well.

Column location has 800249 null values, it means that we have data in more than a million rows. There are 127675 unique locations in this DataFrame. As we can see, the location input wasn't formalized and even when users filled it, they filled it with some creative description of location as "Facing West" or "The Peanut Gallery", which doesn't give precise geographic information. Therefore, despite that the ability to analyze the correlation between tweet sentiment and users' geographic location would be a great idea, unfortunately we cannot do this. The location data wasn't standardized and can only give us information about users' endless creativity.

![img2.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img2.png) 

![img3.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img3.png) 

## RoBERTa Sentiment Analysis

For sentiment analysis, we tried to use the August tweets dataset containing 1050085 rows and 2 columns (datetime and text only). For each tweet, the pretrained RoBERTa model planned to generate a score for each negative, neutral, and positive sentiment.

Unfortunately, the predicted time to complete this task turned out to be 59 hours, and we were forced to stop RoBERTa after 6 hours of work.

![img11.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img11.png) 

However, we have received enough data to our mockup twitter dataset:

![img12.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img12.png) 

## Reducing data

After some research we discovered: there are two popular free Twitter datasets only. Both of them are about 12-13 Gigabytes. All smaller datasets are broken in different ways, Olga tried all of them. We assumed that it is a normal amount of data for this Ukrainian war related tweets, and there is no point for us do our own API downloading, it will be the same size.

How can we reduce the data size?

For example, we can use tweets with only one hashtag. Unfortunately, it definitely will skew our data, hashtags' popularity changes all the time. We're not even sure a certain hashtag existed at Feb 24th! (Most probably it wasn't.)

The same problem is with many other ways of reducing. We don’t know users' gender, age, or geographical position to make a selection according to these parameters. Any selection according to the content of the tweet - quote, retweet, hashtags – skews the data.

But we know one thing for sure about users: the date of their registration. We can take tweets from users who registered before a certain date. 

First, it allows us to meet out technical requirements.
Secondly, we can be sure that there are no fake Kremlin users in this dataset: here is the Kremlin's network of Twitter accounts that work together to retweet and drive up traffic:  

[Kremlin's Network](https://www.bbc.com/news/technology-60790821)

![img7.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img7.png)

Additionally, we can compare our study for the reduced dataset with Alexander Shevtsov's study for the whole dataset. If a significant difference in trends  appears, that means we caught the Kremlin bots working!  

[Alexander Shevtsov study](https://alexdrk14.github.io/RussiaUkraineWar/sentiment.html)  

![img8.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img8.png)

We decided to choose the users that registered on Twitter before 2008. The duration of the war is 179 days to date. Our training dataset is 12 days, i.e. 7%. That means we should be prepared to work with a dataset that is 14 times bigger. The dataset with the tweets from users registered before 2008 has 3266 rows.

Additionally, 2008 is a year when the Russian invasion to Georgia started, and most likely the Kremlin network of bots began to develop.

## Joining the data

For our research, we needed to join the twitter table with the news table. We used PostgreSQL for that:

![img13.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img13.png)

The result is the table with the date, tweets, sentiment of tweets and weight of events:

![img14.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img14.png)

## Description of feature engineering and the feature selection

Our team considered different options: Supervised and Unsupervised learning, Clastering, different Neural Networks for sentiment analysis. 

For sentiment analysis we eventually chose RoBERTa, because it is a free pretrained model from Meta (Facebook), and it is a robustly optimized method for pretraining natural language processing (NLP) systems that improves on Bidirectional Encoder Representations from Transformers, or BERT, the self-supervised method released by Google in 2018. 

For discovering trends and predictions, we chose to use a Supervised Machine Learning Linear Regression Model, considering that as result of prediction we will have negative sentiment, which is a number. 

As we intended to show that Ukrainian War news have an impact on the trend of tweets' negative sentiment, we needed to predict the number of this sentiment for the days in our prediction dataset. Our supervised machine learning model might predict, based on tweet data and information about news, whether a tweet's negative sentiment will rise or not. So far, the machine learning model was able to provide us with labels just for two days of war:

![img15.png](/Preliminary_Data_Analysis/Twitter/Resources/Images/img15.png)

We assume that we can predict the emotional importance of war events by the sentiment analysis of tweets. For the first segment, we used simple linear regression, but at the next segment we could try to use clusterization for Twitter users to obtain a more accurate trend prediction.

Our mockup dataset has data for only two days from 180 days of war to date. This is due to the duration of the calculation of sentiment analysis data: even for those two days it took 6 hours to get the result due to the size of Twitter data.

 
## Technologies, languages, tools, and algorithms used throughout the project:  

**Used tools:**  
beautifulsoup4            4.11.1  
charset-normalizer        2.0.4  
ipykernel                 6.9.1  
importlib-metadata        4.11.3   
huggingface_hub           0.2.1 
hvplot                    0.8.1  
jupyter                   1.0.0   
matplotlib-inline         0.1.2   
numpy                     1.23.1    
pandas                    1.4.3    
prompt_toolkit            3.0.20    
python                    3.10.4   
pytorch                   1.10.2  
sqlite                    3.39.2   
tokenizers                0.11.4  
transformers              4.18.0  
tqdm                      4.64.0   
urllib3                   1.26.11   
scipy                     1.7.3  
scikit-learn              2.2.0  
imbalanced-learn          0.9.0  
ClusterCentroids algorithm  
SMOTEENN algorithm   
RoBERTa (pretrained deep neuron network model)  
pgAdmin                   4.6.8    
postgresql              10.2.16 
Tableau Public 

## Data Visualization 
Link to Tableau dashboard – 
