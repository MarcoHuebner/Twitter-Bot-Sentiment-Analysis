"""
# TODO: Description

"""

import datetime
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from textblob_de import TextBlobDE as TextBlob
# from textblob import TextBlob

from helpers import Helpers


# initialize api
api_helpers = Helpers()

# load and preprocess data
api_helpers.settings()
df = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                              filename="example_search.txt")
trans_df = api_helpers.clean_text_df(df)


# %% Sentiment analysis conditioned on cut-off date
# set cut-off date
cutoff_date = datetime.datetime.strptime('Feb 24 08:00:00 +0000 2021', '%b %d %X %z %Y')
df['cond'] = trans_df['date'] >= cutoff_date

df_past, df_pre = df.loc[df['cond']], df.loc[~df['cond']]

# Sentiment analysis pre and past cutoff_date
sentiment_df_past = api_helpers.sentiment_analysis(df_past)
sentiment_df_pre = api_helpers.sentiment_analysis(df_pre)


# %% plot sentiments (without neutral words)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

sentiment_df_pre.plot.hist(ax=ax1, color='skyblue')
ax1.set_title("Sentiments from Tweets on #NoCovid (pre:" + str(cutoff_date) + ')')
ax1.set_xlabel('sentiment value')

sentiment_df_past.plot.hist(ax=ax2, color='purple')
ax2.set_title("Sentiments from Tweets on #NoCovid (past:" + str(cutoff_date) + ')')
ax2.set_xlabel('sentiment value')
ax2.set_yscale('log')

plt.show()


# %% t test
tstat, pval = stats.ttest_ind(sentiment_df_past['polarity'], sentiment_df_pre['polarity'])
