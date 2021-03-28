# Twitter-Bot & Sentiment-Analysis

### Using the Twitter API for a minor toy project

Up to now this Repo contains some functions regarding **Sentiment analysis** with tweet *regex preprocessing* as form of
tokenization to use simple *bag-of-words*-like methods for *polarity* and *sentiment analysis*. It also supports 
automated *multi-API* search for data generation via [**Tweepy**](https://docs.tweepy.org/en/latest/) - but obviously 
comes without the [API keys](https://developer.twitter.com/en).

Up to now this Repo only contains basic classical machine learning on the data, as without more accurate labelling of
tweets for example RNNs are neither faster and nor more accurate than the baseline, thus making these imo superfluent.

Other useful analyses which are **not included** but might be considered in the future are for example:
- Cluster- and Hashtag-analysis
- [GPS data](https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/)

### Work in progress

- Parameter for language compatibility for [TextBlob](https://textblob.readthedocs.io/en/dev/) (right now only 
  hard-coded)
- Use tqdm for progress tracking when loading tweets
- Still searching for efficient ways to allow search beyond the "recent" week given by twitter's API/ tweepy
- Improve and simplify (for the user) handling of data distribution comparison (MMD vs t-test vs. ...)
- Visually improve plots

### Installation

After cloning, run `pip install -r requirements.txt` to get the relevant packages. Might not include all necessary 
packages yet, but I would like to try that on Linux, not Windows. 

Rename all `config_default.py` to `config_local.py` and specify the correct local path for API-keys, save directory for 
tweets as well as directory for plots to save to.
