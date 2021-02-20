"""
Old toy example, removed from base.py and showing of new data handler as well as basic setup

"""

import tweepy as tw

from helpers import Helpers


# initialize api and settings
api_helpers = Helpers()
api = api_helpers.multi_init_api()

# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'

# Collect tweets with selected API
api_used = 0
tweets = tw.Cursor(api[api_used].search, q=search_words, lang='de', since=data_since).items(5)
df = api_helpers.data_handler(tweets, geo=None, user_metadata=True)
print(df)
