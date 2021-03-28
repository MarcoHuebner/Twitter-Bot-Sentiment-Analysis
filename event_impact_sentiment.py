"""
The following code analyses the sentiment values of tweets. Furthermore, it can
be used to compare sentiments before and after a specific event. Therefore, two
things a necessary: Suitable tweet-data that can be obtained via the procedure
described in example_search.py and a cutoff-date, which marks the event before
and after which you want to compare your tweets.
"""

import datetime
from scipy import stats

from helpers import Helpers


# initialize api
api_helpers = Helpers()

# load and preprocess data
api_helpers.settings()

df1 = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                               filename="lockdown_022621.txt")
df2 = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                               filename="lockdown_030721.txt")
df = df1.append(df2, ignore_index=True)
trans_df = api_helpers.clean_text_df(df)


# %% Sentiment analysis conditioned on cut-off date
# TODO: needs to be automated/ outsourced (e.g. if one file, split according to median, if two files use those separate)
# set cut-off date
print("The median Date of the dataset is ", trans_df['date'].median())
cutoff_date = datetime.datetime.strptime('Feb 24 08:00:00 +0000 2021', '%b %d %X %z %Y')
trans_df['cond'] = trans_df['date'] >= cutoff_date

df_past, df_pre = trans_df.loc[trans_df['cond']], trans_df.loc[~trans_df['cond']]

# Sentiment analysis pre and past cutoff_date on word and tweet_level
sentiment_df_past = api_helpers.sentiment_word_analysis(df_past)
sentiment_df_pre = api_helpers.sentiment_word_analysis(df_pre)

tweet_df_past = api_helpers.sentiment_tweet_analysis(df_past)
tweet_df_pre = api_helpers.sentiment_tweet_analysis(df_pre)

# %% plot sentiments (without neutral words)
# TODO: Improve visualization
api_helpers.plot_sentiment_analysis(sentiment_df_pre, sentiment_df_past, "#NoCovid", cutoff_date, show=True)
api_helpers.plot_sentiment_analysis(tweet_df_pre, tweet_df_past, "#NoCovid", cutoff_date, show=True)

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
