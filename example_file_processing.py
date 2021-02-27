"""
Example for using the new functionality of data_handler using data from tweet_saver from helpers

"""

from helpers import Helpers


# initialize api
api_helpers = Helpers()

# Collect tweets from saved file
df = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                              filename="example_search.txt")
new_df = api_helpers.clean_text_df(df)
print(new_df)
