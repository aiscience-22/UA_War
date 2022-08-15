# War in Ukraine: Sentiment Prediction  
Using Artificial Intellect to discover Twitter trends and Sentiment Analysis
 
We select to discover, how War in Ukraine is presented on Twitter, and to realize Sentiment Prediction for current twits. This subject was selected due to topicality of that information and acute social need for such an analysis.

# Overview of the analysis

The source of data is [Ukraine Conflict Twitter Dataset ](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows?select=0801_UkraineCombinedTweetsDeduped.csv.gzip) with 49.74M tweets on Kaggle. 

![img1.png](/images/img1.png) 

We hope to answer the following question:
* When were the earliest and latest tweets in this dataset created?  
* Visualize august tweet frequency by date  
* How many languages are in this dataset?  
* What percentage of the tweets is in English (en)?
* How many symbols the shortest and longest tweet contains? 
* What is the WordCloud for this dataset?
* What the score for each of negative, neutral, and positive sentiments?
* Which model we should use first for prediction? 
* How we can improve prediction?

# Data exploration phase of the project:

* Description of data preprocessing  
* Description of feature engineering and the feature selection, including the team's decision-making process  
* Description of how data was split into training and testing sets  

**Slides**  
The presentation should be finalized in Google Slides and include the following:  
Slides are primarily images or graphics (rather than primarily text).  
Images are clear, in high-definition, and directly illustrative of subject matter.  

# Analysis phase of the project:

* Explanation of model choice, including limitations and benefits  
* Explanation of changes in model choice (if changes occurred between the Segment 2 and Segment 3 deliverables)  
* Description of how the model was trained (or retrained if the team used an existing model)  
* Description and explanation of model's confusion matrix, including final accuracy score  

**Slides**  
The presentation should be finalized in Google Slides and include the following:  
Slides are primarily images or graphics (rather than primarily text).  
Images are clear, in high-definition, and directly illustrative of subject matter.  

## Technologies, languages, tools, and algorithms used throughout the project:

**Used tools:**
numpy                     1.21.5  
pandas                    1.16.0  
scipy                     1.7.3  
scikit-learn              2.2.0  
imbalanced-learn          0.9.0  
RandomOverSampler  
SMOTE algorithms  
ClusterCentroids algorithm  
SMOTEENN algorithm  
BalancedRandomForestClassifier (bias reduction model)  
EasyEnsembleClassifier (bias reduction model)  

# Result of analysis

* Description and explanation of model's confusion matrix, including final accuracy score

**Live Presentation**  
Requirements for the live presentation follow:  
All team members present in equal proportions.  
The team demonstrates the dashboard's real-time interactivity.  
The presentation falls within any time limits provided by the instructor.  
The submission includes speaker notes, flashcards, or a video of the presentation rehearsal.  

# Recommendation for future analysis

Anything the team would have done differently
