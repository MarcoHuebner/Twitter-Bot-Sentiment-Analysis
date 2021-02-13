# Twitter-Bot & Sentiment-Analysis

### Using the Twitter API for minor (hopefully somewhat) useful toy projects

- Cluster-/ hashtag-analysis as described by 
  [EarthLab](https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/)
  
- Sentiment analysis with text/ regex preprocessing, bag of words etc. 

Should be published (for visibility purposes) after initial jupyter notebooks were removed and the code and repo 
structure is somewhat clean.

### Next steps

- jupyter notebooks will not be functioning (missing data) but probably give some nice code snippets
- additional videos aid explaining stuff from the notebook
- [Tweepy Documentation](https://docs.tweepy.org/en/latest/)
  - [Tweepy API.search Documentation](https://docs.tweepy.org/en/latest/api.html?highlight=api.search#API.search)
  - [Tweepy API Reference](https://docs.tweepy.org/en/latest/api.html) for rate limit handling
- Stuff like [this](https://www.youtube.com/watch?v=W0wWwglE1Vc) might provide further useful first steps.
- Idea: Mining and negative mining 0.1: Try to extract age/ gender etc. data based on tweet history and find ways to 
  "trick" the discriminator by new bot posts to anonymize identity.

### Installation

After cloning, run `pip install -r requirements.txt` to get the relevant packages. Might not include all necessary 
packages yet, but I would like to try that on Linux, not Windows. 

Rename all `config_default.py` to `config_local.py` and specify the correct local paths.