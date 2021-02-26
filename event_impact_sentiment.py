import itertools
import datetime

import matplotlib.pyplot as plt
import pandas as pd

# from textblob import TextBlob
from textblob_de import TextBlobDE as TextBlob
from helpers import Helpers
from scipy import stats

# initialize api
api_helpers = Helpers()


def date_transform(date: str, format: str = '%a %b %d %X %z %Y') -> datetime:
    '''
    transforms date in datetime
    :param date: str, date to be transform into datetime object
    :return: datetime
    '''
    return datetime.datetime.strptime(date, format)


df = api_helpers.data_handler(tweets=None, geo=None, user_metadata=True, from_cursor=False,
                              filename="example_search.txt")
api_helpers.clean_text_df(df)
df['date'] = df['date'].transform(date_transform).sort_values()

cutoff_date = datetime.datetime.strptime('Feb 24 08:00:00 +0000 2021', '%b %d %X %z %Y')
df['cond'] = df['date'] >= cutoff_date

df_past, df_pre = df.loc[df['cond']], df.loc[~df['cond']]

# Sentiment analysis past cutoff_date
sentiment_objects_past = [[TextBlob(word) for word in tweet] for tweet in df_past['text']]
sentiment_values_past = [[[word.sentiment.polarity, str(word)] for word in tweet] for tweet in sentiment_objects_past]
sentiment_df_past = pd.DataFrame(list(itertools.chain(*sentiment_values_past)), columns=["polarity", "tweet"])

# Sentiment analysis pre cutoff_date
sentiment_objects_pre = [[TextBlob(word) for word in tweet] for tweet in df_pre['text']]
sentiment_values_pre = [[[word.sentiment.polarity, str(word)] for word in tweet] for tweet in sentiment_objects_pre]
sentiment_df_pre = pd.DataFrame(list(itertools.chain(*sentiment_values_pre)), columns=["polarity", "tweet"])

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
