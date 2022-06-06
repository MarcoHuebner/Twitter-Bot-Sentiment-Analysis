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
api_helpers.plot_sentiment_analysis(sentiment_df_pre, sentiment_df_past, "#NoCovid", cutoff_date, show=True)
api_helpers.plot_sentiment_analysis(tweet_df_pre, tweet_df_past, "#NoCovid", cutoff_date, show=True)

# NOTE: Before using statistical tests, always check whether their
#       (distribution) requirements are met and examine the raw data
# %% t test to test differences of average polarity score
tstat, pval = stats.ttest_ind(sentiment_df_past['polarity'], sentiment_df_pre['polarity'])
tstat_tweet, pval_tweet = stats.ttest_ind(tweet_df_past['polarity'], tweet_df_pre['polarity'])

print("Two-sided t-test results, p<0.05 is usually used as threshold for significance of a difference.")
print("Two-sided t-test p-value of the word polarity: ", pval)
print("Two-sided t-test p-value of the tweet polarity: ", pval_tweet)

# %% Kolmogorov-Smirnov test to test differences of the distributions
stat, val = stats.kstest(sentiment_df_past['polarity'], sentiment_df_pre['polarity'], alternative="two-sided",
                         mode="exact")
stat_tweet, val_tweet = stats.kstest(tweet_df_past['polarity'], tweet_df_pre['polarity'], alternative="two-sided",
                                     mode="exact")

print("Two-sided KS-test results, p<0.05 is usually used as threshold for significance of a difference.")
print("Two-sided KS-test p-value of the word polarity: ", val)
print("Two-sided KS-test p-value of the tweet polarity: ", val_tweet)

# %% MMD to check similarity between the tweet distributions, the divergence score needs more careful interpretation (!)
print("MMD with 'multiscale' kernel result:", api_helpers.mmd(sentiment_df_past['polarity'].values,
                                                              sentiment_df_pre['polarity'].values, kernel="multiscale"))
print("MMD with 'rbf' kernel result: ", api_helpers.mmd(sentiment_df_past['polarity'].values,
                                                        sentiment_df_pre['polarity'].values, kernel="rbf"))
