
# coding: utf-8

# # Twitter Sentiment Analysis
# 
# Extract tweets about a particular topic from twitter (recency = 1-7 days) and analyze the opinion of tweeples (people who use twitter.com) on this topic as positive, negative or neutral. 
# 
# ## What is sentiment analysis?
# 
# Process of ‘computationally’ determining whether a piece of writing is positive, negative or neutral. It’s also known as opinion mining, deriving the opinion or attitude of a speaker.
# 
# ## Why sentiment analysis?
# 
# ### Business: 
# 
# In marketing field companies use it to develop their strategies, to understand customers’ feelings towards products or brand, how people respond to their campaigns or product launches and why consumers don’t buy some
# products.
# ### Politics: 
# In political field, it is used to keep track of political view, to detect consistency and inconsistency between statements and actions at the government level. It can be used to predict election results as well!
# ### Public Actions: 
# Sentiment analysis also is used to monitor and analyse social phenomena, for the spotting of potentially dangerous situations and determining the general mood of the blogosphere.
# 
# ## Steps involved in Sentiment Analysis 
# 
# ### Traininig
# 
# <img src="training.png">
# 
# ### Prediction
# 
# <img src="pred.png">
# 
# ---
# 
# ## Training the Classifiers
# 
# The classifiers need to be trained and to do that, we need to list manually classified tweets. Let's start with 3 positive, 3 neutral and 3 negative tweets.
# 
# ### Preprocess tweets
# 1. Lower Case - Convert the tweets to lower case.
# 2. URLs - eliminate all of these URLs via regular expression matching or replace with generic word URL.
# 3. @username - we can eliminate "@username" via regex matching or replace it with generic word AT_USER.
# 4. #hashtag - hash tags can give us some useful information, so it is useful to replace them with the exact same word without the hash. E.g. #nike replaced with 'nike'.
# 5. Punctuations and additional white spaces - remove punctuation at the start and ending of the tweets. E.g: ' the day is beautiful! ' replaced with 'the day is beautiful'. It is also helpful to replace multiple whitespaces with a single whitespace
# 

# In[1]:


import os
os.getcwd()


# In[9]:


#import regex
import re

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    #  #NIKE ---> NIKE
    #tweet = re.sub(r'#([^\s]+)', r'HASH\1', tweet)
    tweet = re.sub(r'[\.!:\?\-\'\"\\/]', r'', tweet)
    #trim
    
    tweet = tweet.strip('\'"')
    return tweet
#end


# In[11]:



#Read the tweets one by one and process it
fp = open('Files/SampleTweets.txt', 'r') # r -> read w -> write rw ->
line = fp.readline() # identify a line based on \n

while line:
    processedTweet = processTweet(line)
    print(processedTweet)
    line = fp.readline() # reads till EOF
#end loop
fp.close()


# ### Feature Vector
# 
# A good feature vector directly determines how successful your classifier will be. 
# 
# The feature vector is used to build a model which the classifier learns from the training data and further can be used to classify previously unseen data.
# 
# We can use the presence/absence of words that appear in tweet as features. 
# 
# In the training data, consisting of positive, negative and neutral tweets, we can split each tweet into words and add each word to the feature vector. 
# 
# Some of the words might not have any say in indicating the sentiment of a tweet and hence we can filter them out. 
# 
# Adding individual (single) words to the feature vector is referred to as 'unigrams' approach.
# 
# Some of the other feature vectors also add 'bi-grams' in combination with 'unigrams'. For example, 'not good' (bigram) completely changes the sentiment compared to adding 'not' and 'good' individually. 
# 
# Here, for simplicity, we will only consider the unigrams. Before adding the words to the feature vector, we need to preprocess them in order to filter, otherwise, the feature vector will explode.
# 
# #### Filtering tweet words (for feature vector)
# 
# 1. Stop words - a, is, the, with etc. The full list of stop words can be found at Stop Word List. These words don't indicate any sentiment and can be removed.
# 2. Repeating letters - if you look at the tweets, sometimes people repeat letters to stress the emotion. E.g. hunggrryyy, huuuuuuungry for 'hungry'. We can look for 2 or more repetitive letters in words and replace them by 2 of the same.
# 3. Punctuation - we can remove punctuation such as comma, single/double quote, question marks at the start and end of each word. E.g. beautiful!!!!!! replaced with beautiful
# 

# In[15]:


#initialize stopWords
stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet,stopWords):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return list(set(featureVector))
#end

#Read the tweets one by one and process it
fp = open('Files/SampleTweets.txt', 'r')
line = fp.readline()

st = open('Files/StopWords.txt', 'r')
stopWords = getStopWordList('Files/StopWords.txt')

while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet,stopWords)
    print(featureVector)
    line = fp.readline()
#end loop
fp.close()


# In[16]:


# For a bigger training dataset
import csv
#Read the tweets one by one and process it
inpTweets = csv.reader(open('Files/SampleTrainingData.csv', 'r'), delimiter=',')
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet,stopWords)
    tweets.append((featureVector, sentiment));
#end loop


# In[20]:


tweets[15000:15010]


# In[21]:


#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


# #### Bulk Extraction of Features

# In[22]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
#from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


# In[23]:


#Read the tweets one by one and process it
inpTweets = csv.reader(open('Files/SampleTrainingData.csv', 'r'), delimiter=',', quotechar='|')
stopWords = getStopWordList('Files/StopWords.txt')
featureList = []

# Get tweet words
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, tweets)


# In[25]:


#This is a feature vector

# The first step in creating a classifier is deciding what features of the input are relevant, 
# and how to encode those features.

# The returned dictionaries for all tweets, known as a feature set, 
# maps from feature names to their values.

# does a tweet contains(a certain feature word (pos or neg)): True or False

# Feature Words or the complete BoW
# This BoW is a mix of pos and neg words
# In a neg tweet : I'll check for each and every word in the BoW

training_set[2] # neg

"pathetic service and  bad fod"
# contains("awesome") : False
# contains("ambience") : False
# contains("awareness") : False
# contains("beautiful") : False
# ..
# contains("bad") : True
# contains("fist") : False
# contains("fillet") : False
# contains("fever") : False
# contains("food") : True


# ## Classifier Algorithm : Naive Bayes Classifier
# 
# At this point, we have a training set, so all we need to do is instantiate a classifier and classify test tweets. The below code explains how to classify a single tweet using the classifier.
# 
# Uses Bayes theorem of probability to predict the class of unknown data set.
# 
# P(A|B) = (P(B|A).P(A))/P(B)
# 
# Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
# 
# 
#Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)#pickle the above file
import pickle

pickle_out = open("NBC_Save.pickle","wb")
pickle.dump(NBClassifier, pickle_out)
pickle_out.close()
# In[27]:


l1 = [12,21,54,2365,323,723,624,624,73]
import pickle

pickle_out = open("list_l1_save.pickle","wb")
pickle.dump(l1, pickle_out)
pickle_out.close()


# In[28]:


l1


# In[29]:


del l1


# In[31]:


l1


# In[32]:


pickle_in = open("list_l1_save.pickle","rb")
restored_l1 = pickle.load(pickle_in)


# In[33]:


restored_l1


# In[36]:


# Load when required
import pickle
pickle_in = open("NBC_Save.pickle","rb")
NBClassifier = pickle.load(pickle_in)


# In[40]:


test_tweet = "I am so glad to use this service. Extremely satisfied and happy with @gateway hotel"


# In[41]:


processedTestTweet = processTweet(test_tweet)
processedTestTweet
feature_words = extract_features(getFeatureVector(processedTestTweet,stopWords))
feature_words


# In[42]:


NBClassifier.classify(feature_words) # testing


# In[43]:


testTweet = "pathetic service by @jetairways. Seat belts aren't proper!"
processedTestTweet = processTweet(testTweet)
print(NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords))))


# In[ ]:


print(NBClassifier.show_most_informative_features(10))


# In[44]:


testTweet = 'Pathetic staff, worse service. Never flying with #AirIndianaJones'
processedTestTweet = processTweet(testTweet)
print(NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords))))


# In[53]:


test = "I bought onepluse6t product on 1st Nov and today my phone cameras are not working . Please consider my experience with OnePlus before you buying this product."
processedTestTweet = processTweet(test)
print(NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords))))


# In[52]:


test = "Wonderful product, so happy and blissful with something that stops working after the first day! :|"
processedTestTweet = processTweet(test)
print(NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords))))


# ---
# #### Getting our test data from Twitter by connecting to it via API 
# config.json

{
    "consumer_key": "YOUR CONSUMER KEY",
    "consumer_secret": "YOUR CONSUMER SECRET",
    "access_token": "YOUR ACCESS TOKEN",
    "access_token_secret": "YOUR ACCESS TOKEN SECRET"
}https://developer.twitter.com/en/docs/api-reference-index

# How to connect?
https://auth0.com/docs/connections/social/twitterpip install twython
# tweepy
# In[41]:


from twython import Twython  
import json

# Load credentials from json file
with open("TwitterAPI/config.json", "r") as file:  
    creds = json.load(file)

# Instantiate an object
python_tweets = Twython(creds['consumer_key'], creds['consumer_secret'],creds['access_token'], creds['access_token_secret'])

# Create our query
query = {'q': 'ThursdayThoughts',  
        'result_type': 'popular',
        'count': 10000,
        'lang': 'en',
        }


# In[48]:


import pandas as pd

# Search tweets
dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}  
for status in python_tweets.search(**query)['statuses']:  
    dict_['user'].append(status['user']['screen_name'])
    dict_['date'].append(status['created_at'])
    dict_['text'].append(status['text'])
    dict_['favorite_count'].append(status['favorite_count'])

# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)  
df.head(5) 


# In[49]:


# Or read a timeline
twitter =Twython()

user_timeline = python_tweets.get_user_timeline(screen_name="narendramodi")
# And print the tweets for that user.
pm_modi = []
for tweet in user_timeline:
    pm_modi.append(tweet['text'])


# In[50]:


sentiment = []
for test in pm_modi:
    processedTestTweet = processTweet(test)
    feature_words = extract_features(getFeatureVector(processedTestTweet,stopWords))
    sentiment.append(NBClassifier.classify(feature_words))


# In[51]:


pd.concat([pd.Series(pm_modi),pd.Series(sentiment)],axis=1,keys=["tweet","sentiment"])


# In[52]:


from textblob import TextBlob


# In[54]:


TextBlob("Pawan in the class leader").detect_language()


# In[55]:


TextBlob("Pawan ist der klassen fuhrer").detect_language()


# In[56]:


TextBlob("bahar mausam bada acha hai").detect_language()


# In[ ]:


TextBlob.translator.translate


# In[61]:


TextBlob.translator.translate("Pawan ist der klassen fuhrer")

# Further Tasks

# 1. Properly clean the training data (SampleTrainingData.csv)
# 2. Ignore non English text when training
#     Hint : nltk can identify the language
             Also, it can transalate the words
# 3. Control the twitter feed (for validation dataset):
#     a. No of tweets
#     b. Time line for the tweets
#     c. get tweets for a topic only from a geographical area (topic : OnePlus, then get tweets from India)
# the nltk has certain functions to ID
#     d. Identify the language used in the tweets
#     e. Remove/translate non-english tweets

# In[ ]:


# 1. Email classification LDA
#    document classification github

# 2. Text scrubbing

# 3. Log file extraction automation

