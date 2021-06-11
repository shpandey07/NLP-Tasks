#!/usr/bin/env python
# coding: utf-8

# EX4.2- Word Sense Disambiguation with NLTK
# 
# Use the Lesk algorithm provided by Python NLTK Word Sense Disambiguation (WSD)
#     module to find the correct definition (i.e. WordNet’s synset definition) of the word
#     “rock” in the following sentences:
#     
# ➢ “A rock is classified according to characteristics such as mineral and chemical composition”;
# 
# ➢ “Queen are a British rock band formed in London in 1970”.

# - Sentence1 :“A rock is classified according to characteristics such as mineral and chemical composition”

# In[24]:


from nltk.corpus import wordnet as wn
for ss in wn.synsets('rock'):
     print(ss, ss.definition())      #print definitions of rock from wordnet


# In[25]:


from nltk.wsd import lesk
#tokenize
sent = ['A', 'rock', 'is', 'classified', 'according', 'to', 'characteristics', 'such', 'as', 'mineral', 'and', 'chemical', 'composition',]
print(lesk(sent, 'rock', 'n')) #print correct definition of rock for the given sentence
print(lesk(sent, 'rock'))


# Synset('rock.n.04')-(figurative) someone who is strong and stable and dependable; ; --Gospel According to Matthew: is the correct definition of the word 'rock' in the first sentence. 

# - Sentence2: “Queen are a British rock band formed in London in 1970”.

# In[26]:


from nltk.corpus import wordnet as wn
for ss in wn.synsets('rock'):
     print(ss, ss.definition())     #print definitions of rock from wordnet


# In[27]:


from nltk.wsd import lesk
sent = ['Queen', 'are', 'a', 'British', 'rock', 'band', 'formed', 'in', 'London', 'in', '1970'] #tokenize
print(lesk(sent, 'rock', 'n')) #print correct definition of rock for the given sentence
print(lesk(sent, 'rock'))


# Synset('rock_'n'_roll.n.01')-  a genre of popular music originating in the 1950s; a blend of black rhythm-and-blues with white country-and-western is the correct definition of the word 'rock' in the second sentence.

# ______________________________________________________________________________________________________________________________

# EX4.3- Sentiment Analysis with NLTK
# 
# ❖ Perform Sentiment Analysis on tweets following the steps listed below:
# 
# ➢ Get the dataset in the “Exercise resources” folder. The given dataset contains data about the
# tweet: IDs, timestamp, topic, author username, text;
# 
# ➢ Classify the sentiment of tweets for “Nike” topic using NLTK’s Liu and Hu opinion lexicon.
# Compute their distribution (e.g. 70% positive, 10% negative, 20% neutral);
# 
# ➢ Plot the visual representation of the sentence polarity for the “World Cup 2010” tweet using the
# same Liu and Hu opinion lexicon;
# 
# ➢ Compute the average “compound” polarity scores for “dentist” topic using NLTK’s Vader
# approach.

# (i) Classify the sentiment of tweets for “Nike” topic using NLTK’s Liu and Hu opinion lexicon.
# Compute their distribution (e.g. 70% positive, 10% negative, 20% neutral)

# Step 1: Data Preprocessing

# In[28]:


#import the required libraries
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
# import ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# Global Parameters
stop_words = set(stopwords.words('english'))


# In[29]:


sentiment_df = pd.read_csv('C:/Users/Dell/Desktop/SoSe21/KEDH/Exercise/Ex4/dataset_sentiment_analysis.csv',header=None, names=['polarity','id', 'timestamp', 'topic', 'author', 'text'])
sentiment_df.head()


# In[30]:


sentiment_df.head()


# In[31]:


sentiment_df.shape #check dimensionality of data


# In[32]:


sentiment_df.info() #summary of data


# In[33]:


def preprocess_tweet_text(tweet):
    tweet.lower() #convert text to lower case
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    #lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(filtered_words)
    


# In[34]:


sentiment_df.text = sentiment_df['text'].apply(preprocess_tweet_text)


# In[35]:


sentiment_df.head()


# In[36]:


sentiment_df2 = sentiment_df[(sentiment_df.topic == "nike")]
sentiment_df2.head()


# In[37]:


#vectorization
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

#splitting the data
tf_vector = get_feature_vector(np.array(sentiment_df2.iloc[:, 4]).ravel())
X = tf_vector.transform(np.array(sentiment_df2.iloc[:, 4]).ravel())
y = np.array(sentiment_df2.iloc[:, 1]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)


# In[38]:


#using Liu and Hu opinion lexicon to categorize results into positive, negative and neutral

from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus.reader.api import *


class IgnoreReadmeCorpusView(StreamBackedCorpusView):
    """
    This CorpusView is used to skip the initial readme block of the corpus.
    """

    def __init__(self, *args, **kwargs):
        StreamBackedCorpusView.__init__(self, *args, **kwargs)
        # open self._stream
        self._open()
        # skip the readme block
        read_blankline_block(self._stream)
        # Set the initial position to the current stream position
        self._filepos = [self._stream.tell()]
        

class OpinionLexiconCorpusReader(WordListCorpusReader):
    
    CorpusView = IgnoreReadmeCorpusView

    def words(self, fileids=None):
        """
        Return all words in the opinion lexicon. Note that these words are not
        sorted in alphabetical order.

        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat(
            [
                self.CorpusView(path, self._read_word_block, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ]
        )


    def positive(self):
        """
        Return all positive words in alphabetical order.
        
        """
        return self.words("positive-words.txt")


    def negative(self):
        """
        Return all negative words in alphabetical order.

        """
        return self.words("negative-words.txt")


    def _read_word_block(self, stream):
        words = []
        for i in range(20):  # Read 20 lines at a time.
            line = stream.readline()
            if not line:
                continue
            words.append(line.strip())
        return words


# In[ ]:




