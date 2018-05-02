
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk import ngrams


# In[2]:


#input gold price dataset and convert it into time series data
rawPriceData=pd.read_csv('/Users/Mandy/Study/SpringBoard/Capstone 2/GoldPriceData-Raw.csv')
rawPriceData['Date'] = pd.to_datetime(rawPriceData['Date'])
rawPriceData.set_index(rawPriceData['Date'],inplace=True)


# In[3]:


# filling weekend/public holiday missing data by taking average.
priceData=rawPriceData.resample('D').interpolate()
priceData=priceData['Price']


# In[4]:


#input tweet dataset
rawTweetData=pd.read_csv('/Users/Mandy/Study/SpringBoard/Capstone 2/TweetData-Raw.csv',encoding='latin1')
print(rawTweetData.head())


# In[5]:


#replace special characters, such as URLs, usernames, hashtags, pictures with"URL","USER","HASHTAG","PICTURE"
#also remove all the special characters
for row in rawTweetData.index:
    eachTweet=rawTweetData.Tweet[row]
    eachTweet_url=re.sub(r"https:// \S+","URL",eachTweet)
    eachTweet_pic=re.sub(r"pic.\S+","PICTURE",eachTweet_url)
    eachTweet_user=re.sub(r"@\S+","USER",eachTweet_pic)
    eachTweet_hashtag=re.sub(r"#\S+","HASHTAG",eachTweet_user)
    eachTweet_spechar=re.sub("[^A-Za-z0-9]+"," ",eachTweet_hashtag)
    rawTweetData.set_value(row,'Tweet',eachTweet_spechar)
print(rawTweetData.Tweet[0])
print(rawTweetData.Tweet[1])
print(rawTweetData.Tweet[2])
print(rawTweetData.Tweet[3])
print(rawTweetData.Tweet[4])


# In[6]:


#Split tweet text word by word (tokenize)
tokenizedTweet = rawTweetData.apply(lambda row: word_tokenize(row['Tweet']), axis=1)
print(tokenizedTweet.head())


# In[9]:


#remove stopwords( words that don't have any positive/negative meanings)
filteredTweet = tokenizedTweet.apply(lambda x: [word for word in x if word not in stopwords.words('english')])
print(filteredTweet.head())


# In[17]:


testing=filteredTweet[0]
print(testing)
trigrams=ngrams(testing,3)
for g in trigrams:
    print(g)

