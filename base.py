import os
import tweepy as tw
import pandas as pd

from helpers import Helpers

# initialize api
api = Helpers.init_api('earth_data_science/keys/.keys.csv')

# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'

# Collect tweets
tweets = tw.Cursor(api.search, q=search_words, lang='de', since=data_since).items(5)

# write user name, location and full text into DataFrame
print(Helpers.data_handler(tweets, info=["user", "location", "full_text"]))