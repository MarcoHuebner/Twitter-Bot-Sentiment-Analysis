# %%
import os
import itertools
import collections

import networkx as nx
import tweepy as tw
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk import bigrams
from helpers import Helpers
from config_local import ConfigPaths
from textblob import TextBlob

# set working directory  # TODO: What does this do? If needed in your IDE
# consider putting it in config default/ local
os.chdir(ConfigPaths().work_dir)

# initialize api and settings
api_helpers = Helpers()
api = api_helpers.init_api()
api_helpers.settings(warning="ignore")


# ################ Old toy example ############################################
# %%
# Define search terms
search_words = '#lockdown' + ' -filter:retweets'
data_since = '2021-02-11'

# Collect tweets TODO: Automate tw.Cursor in separate function
tweets = tw.Cursor(api.search, q=search_words, lang='de', since=data_since).items(5)
df = api_helpers.data_handler(tweets, info=["user", "location", "full_text"])
print(df)
###############################################################################

# %%
# Tweet word frequency analysis
search_words = '#climate+change -filter:retweets'
stop_words = stopwords.words('english')
collection_words = ['climate', 'change', 'climatechange']

# TODO: Make a general text preprocessing function, incorporating all these steps
tweets = tw.Cursor(api.search, q=search_words, lang='en', since=data_since).items(100)
df = api_helpers.data_handler(tweets, info=["user", "location", "full_text"])
df = api_helpers.get_words(df, collection_words, stop_words)
df

# %%

clean_tweets_nsw = pd.DataFrame(list(itertools.chain(*df['full_text'])), columns=["words"])\
    .value_counts().rename("count").to_frame().head(15)

# %%
# Plot
fig, ax = plt.subplots(figsize=(8, 8))
clean_tweets_nsw.plot.barh(y='count', ax=ax, color='purple')
ax.set_title('Common Words Found in Tweets')
plt.show()

# %%
# Create bigrams and split them into letters (from only one row of df...?)
terms_bigram = [list(bigrams(tweet)) for tweet in df['full_text']]
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)
bigram_df = pd.DataFrame(bigram_counts.most_common(20), columns=['bigram', 'count'])

# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')

# %% Create network plot
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

G.add_node("china", weight=100)
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, k=2)

# Plot networks
nx.draw_networkx(G, pos, font_size=16, width=3, edge_color='grey',
                 node_color='purple', with_labels=False, ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y, s=key, bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
plt.show()

# %%

# Create textblob objects of the tweets
sentiment_objects = [[TextBlob(word) for word in tweet] for tweet in df['full_text']]

# Create df of polarity valuesx and tweet text
sentiment_values = [[[word.sentiment.polarity, str(word)]
                     for word in tweet] for tweet in sentiment_objects]
sentiment_df = pd.DataFrame(list(itertools.chain(*sentiment_values)), columns=["polarity", "tweet"])
sentiment_df.sort_values('polarity')

# plot sentiments (without neutral words)
fig, ax = plt.subplots(figsize=(8, 8))
sentiment_df[sentiment_df.polarity != 0].plot.hist(ax=ax, color='purple')
plt.title("Sentiments from Tweets on Climate Change")
plt.show()
