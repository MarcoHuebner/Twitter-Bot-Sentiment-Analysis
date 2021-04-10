"""
The following code analyses the sentiment values of tweets. Furthermore, it can
be used to compare sentiments before and after a specific event. Therefore, two
things a necessary: Suitable tweet-data that can be obtained via the procedure
described in examples/example_search.py and a cutoff-date, which marks the event
before and after which you want to compare your tweets.

"""

import datetime
from scipy import stats

from helpers import Helpers


# initialize api
api_helpers = Helpers()

# load and preprocess generated data
api_helpers.settings()

df1 = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                               filename="lockdown_022621.txt")
df2 = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                               filename="lockdown_030721.txt")

# set cut-off date and clean & split data
cutoff_date = datetime.datetime.strptime('Feb 24 08:00:00 +0000 2021', '%b %d %X %z %Y')

df = df1.append(df2, ignore_index=True)
df_past, df_pre = api_helpers.split_df(df, cutoff_date=cutoff_date)

# Sentiment analysis before and past cutoff_date on word and tweet level
sentiment_df_past = api_helpers.sentiment_word_analysis(df_past)
sentiment_df_pre = api_helpers.sentiment_word_analysis(df_pre)

tweet_df_past = api_helpers.sentiment_tweet_analysis(df_past)
tweet_df_pre = api_helpers.sentiment_tweet_analysis(df_pre)


# %% plot sentiments (without neutral words)
# TODO: Improve visualization
api_helpers.plot_sentiment_analysis(sentiment_df_pre, sentiment_df_past, cutoff_date, title="#NoCovid", show=True)
api_helpers.plot_sentiment_analysis(tweet_df_pre, tweet_df_past, cutoff_date,  title="#NoCovid", show=True)

# TODO: think on own, small statistic module for easy to interpret stat. comparison
# %% t test
tstat, pval = stats.ttest_ind(sentiment_df_past['polarity'], sentiment_df_pre['polarity'])
tstat_tweet, pval_tweet = stats.ttest_ind(tweet_df_past['polarity'], tweet_df_pre['polarity'])

print("Two-sided t-test result on the word polarity analysis: ", tstat, pval)
print("Two-sided t-test result on the tweet polarity analysis: ", tstat_tweet, pval_tweet)

# %% MMD to check similarity between the tweet distributions
print("MMD with 'multiscale' kernel result:", api_helpers.mmd(sentiment_df_past['polarity'].values,
                                                              sentiment_df_pre['polarity'].values, kernel="multiscale"))
print("MMD with 'rbf' kernel result: ", api_helpers.mmd(sentiment_df_past['polarity'].values,
                                                        sentiment_df_pre['polarity'].values, kernel="rbf"))
