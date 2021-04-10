"""
Toy example showing of data handler as well as basic setup with tweepy (and without our methods).

"""

import tweepy as tw

from helpers import Helpers


# initialize api
api_helpers = Helpers()
api = api_helpers._multi_init_api()

# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'
lang = 'de'
items = 10

# Collect tweets with first API
api_used = 0
tweets = tw.Cursor(api[api_used].search, q=search_words, lang=lang, since=data_since).items(items)
df = api_helpers.data_handler(tweets, geo=None, user_metadata=True)
print(df)
