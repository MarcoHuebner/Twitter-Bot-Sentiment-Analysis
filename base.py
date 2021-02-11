import os
import tweepy as tw
import pandas as pd


keys = pd.read_csv('earth_data_science/keys/.keys.csv')
keys.at[1, 'consumer_key']


auth = tw.OAuthHandler(keys.at[0, 'consumer_key'], keys.at[0, 'consumer_secret'])
auth.set_access_token(keys.at[0, 'access_token'], keys.at[0, 'access_token_secret'])
api = tw.API(auth, wait_on_rate_limit=True)

# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'

# Collect tweets
tweets = tw.Cursor(api.search,
                   q=search_words,
                   lang='de',
                   since=data_since).items(5)
[tweet.text for tweet in tweets]
users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
tweet_text = pd.DataFrame(data=users_locs, columns=['user', 'location'])
tweet_text