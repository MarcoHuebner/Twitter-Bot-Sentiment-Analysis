"""
Example for using the new tweet_saver and cursor_search from helpers

"""

from helpers import Helpers


# initialize api
api_helpers = Helpers()

# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'
lang = 'de'
items = 10

# save tweets
api_helpers.tweet_saver(filename="example_search.txt", search_words=search_words, lang=lang, items=items)
