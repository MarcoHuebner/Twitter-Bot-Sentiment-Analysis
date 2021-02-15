# %%
import os
import nltk
import itertools
import collections
import re
import networkx
import warnings

import tweepy as tw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from helpers import Helpers
from config_local import ConfigPaths

# %%
# settings
warnings.filterwarnings('ignore')

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

# set working directory
os.chdir(ConfigPaths().work_dir)

# initialize api
api_helpers = Helpers()
api = api_helpers.multi_init_api()

# %%
# Functions
def remove_url(txt):
    return ' '.join(re.sub('([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', txt).split())

# %%
# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'

# Collect tweets
tweets = tw.Cursor(api.search, q=search_words, lang='de', since=data_since).items(5)
print(api_helpers.data_handler(tweets, info=["user", "location", "full_text"]))

# %%
# Tweet word frequency analysis
search_words = '#climate+change -filter:retweets'

tweets = tw.Cursor(api.search, q=search_words, lang='en', since=data_since).items(1000)

all_tweets = [tweet.text for tweet in tweets]
all_tweets[:5]

all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
word_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]

# %%
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in word_in_tweet]

all_words_nsw = list(itertools.chain(*tweets_nsw))
counts_nsw = collections.Counter(all_words_nsw)




clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(15),
                                    columns=['words', 'count'])

# %%
# Plot
fig, ax = plt.subplots(figsize=(8, 8))
clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
                                                       y='count',
                                                       ax=ax,
                                                       color='purple')
ax.set_title('Common Words Found in Tweets')
plt.show()

# %%
collection_words = ['climate', 'change', 'climatechange']

tweets_nsw_nc = [[word for word in tweet_words if not word in stop_words or collection_words]
                 for tweet_words in word_in_tweet]
tweets_nsw_nc[0]
# %%
