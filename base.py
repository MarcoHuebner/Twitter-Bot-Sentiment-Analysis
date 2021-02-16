# %%
import os
import itertools
# import collections
# import networkx # TODO

import tweepy as tw
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from helpers import Helpers
from config_local import ConfigPaths

# set working directory  # TODO: What does this do? If needed in your IDE consider putting it in config default/ local
os.chdir(ConfigPaths().work_dir)

# initialize api and settings
api_helpers = Helpers()
api = api_helpers.multi_init_api()[0]  # only use first key
api_helpers.settings(warning="ignore")

# ################ Old toy example #####################################################################################
# %%
# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'

# Collect tweets # TODO: Automate tw.Cursor in separate function
tweets = tw.Cursor(api.search, q=search_words, lang='de', since=data_since).items(5)
print(api_helpers.data_handler(tweets, info=["user", "location", "full_text"]))
########################################################################################################################

# %%
# Tweet word frequency analysis
search_words = '#climate+change -filter:retweets'

# TODO: Make a general text preprocessing function, incorporating all these steps
tweets = tw.Cursor(api.search, q=search_words, lang='en', since=data_since).items(1000)
all_tweets = [tweet.text for tweet in tweets]
all_tweets_no_urls = [api_helpers.remove_url(tweet) for tweet in all_tweets]
word_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]

# %%
stop_words = stopwords.words('english')
tweets_nsw = [[word for word in tweet_words if word not in stop_words] for tweet_words in word_in_tweet]
clean_tweets_nsw = pd.DataFrame(list(itertools.chain(*tweets_nsw)), columns=["words"])\
                   .value_counts().rename("count").to_frame().head(15)

# %%
# Plot
fig, ax = plt.subplots(figsize=(8, 8))
clean_tweets_nsw.plot.barh(y='count', ax=ax, color='purple')
ax.set_title('Common Words Found in Tweets')
plt.show()

# %%
collection_words = ['climate', 'change', 'climatechange']

tweets_nsw_nc = [[word for word in tweet_words if word not in stop_words or collection_words]
                 for tweet_words in word_in_tweet]
print(tweets_nsw_nc[0])
# %%
