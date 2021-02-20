"""
Provides helper functions to call in (main) script

"""

import os
import re
import nltk
import json
import warnings
import numpy as np
import tweepy as tw
import pandas as pd
import seaborn as sns
from typing import Any, List, Union
from config_local import ConfigPaths


class Helpers(object):
    def __init__(self):
        pass

    @staticmethod
    def settings(warning: str = None) -> None:
        """
        One common place for common settings.
        :param warning: str, standard setting is None, allowed parameters given by warnings.filterwarnings, e.g. ignore
        :return: None
        """
        # set relative working directory
        os.chdir(ConfigPaths().work_dir)
        # ignore warnings if warning string not None
        if warning:
            warnings.filterwarnings('ignore')
        # nltk stopwords download
        nltk.download('stopwords')
        # plot settings
        sns.set(font_scale=1.5)
        sns.set_style('whitegrid')
        return None

    @staticmethod
    def init_api(row: int = 0) -> Any:
        """
        Sets tokens and returns set up API.
        :param row: int, row number of key set, default 0. Added for compatibility with multiple accounts/ keys
        :return: tw.API, initialized to be waiting on rate limit
        """
        # exception handling: Existence and emptiness of given directory
        if os.path.isfile(ConfigPaths().key_dir):
            if os.stat(ConfigPaths().key_dir).st_size != 0:
                keys = pd.read_csv(ConfigPaths().key_dir)
                auth = tw.OAuthHandler(keys.at[row, 'consumer_key'],
                                       keys.at[row, 'consumer_secret'])
                auth.set_access_token(keys.at[row, 'access_token'],
                                      keys.at[row, 'access_token_secret'])
                return tw.API(auth, wait_on_rate_limit=True)
        else:
            raise IOError("Key directory non-existent or empty. Check the README on renaming config_default.py and "
                          "check for correctness of given path")

    def multi_init_api(self) -> Union[List[Any]]:
        """
        Sets tokens and returns set up API or list of APIs.
        :return: Union[List[tw.API]], initialized to be waiting on rate limit
        """
        # exception handling: Existence and emptiness of given directory
        if os.path.isfile(ConfigPaths().key_dir):
            if os.stat(ConfigPaths().key_dir).st_size != 0:
                # check number of keys/ tokens
                keys = pd.read_csv(ConfigPaths().key_dir)
                if len(keys) > 1:
                    api_list = []
                    for index, row in keys.iterrows():
                        api_list.append(self.init_api(row=index))

                    return api_list
                else:
                    return list(self.init_api())
        else:
            raise IOError("Key directory non-existent or empty. Check the "
                          "README on renaming config_default.py and check for "
                          "correctness of given path")

    @staticmethod
    def data_handler(tweets: Any, geo: bool = None, user_metadata: bool = True) -> pd.DataFrame:
        """
        # TODO: Simplify and accelerate the function

        :param tweets: tw.Cursor search, delivering filtered and downloaded tweets # TODO: read this from file where pkl
                                                                                   # TODO: dumped _json
        :param geo: bool, filter for available geo data and skip if no data given
        :param user_metadata: bool, determine whether to collect user metadata like followers count or friends count as
                              well as mentioned users
        :return: pd.DataFrame, containing only the given/ relevant columns
        """
        # set up empty list to append row_dict to (faster than DataFrame.append)
        row_list = []

        # Process data from tweets
        for tweet in tweets:
            # check for geo values if desired, skip tweet if unsuccessful
            if geo:
                if tweet.geo is None:
                    continue

            # load data from json or through tweepy methods
            tweet_json = json.dumps(tweet._json)

            row_dict = {"text": tweet.text}
            hashtags = []
            for i, value in enumerate(tweet.entities['hashtags']):
                hashtags.append(tweet.entities['hashtags'][i]["text"])
            row_dict.update({"hashtags": hashtags})
            row_dict.update({"source": tweet.source})
            row_dict.update({"user_id": json.loads(tweet_json)["user"]["id"]})
            row_dict.update({"user_screen_name": json.loads(tweet_json)["user"]["screen_name"]})
            row_dict.update({"user_name": json.loads(tweet_json)["user"]["name"]})
            row_dict.update({"location": json.loads(tweet_json)["user"]["location"]})
            row_dict.update({"description": json.loads(tweet_json)["user"]["description"]})
            row_dict.update({"protected": json.loads(tweet_json)["user"]["protected"]})
            row_dict.update({"coordinates": tweet.coordinates})
            row_dict.update({"retweet_count": json.loads(tweet_json)["retweet_count"]})
            row_dict.update({"favourite_count": json.loads(tweet_json)["favorite_count"]})
            # TODO: KeyError: 'possibly_sensitive'
            # possibly_sensitive = json.loads(tweet_json)["possibly_sensitive"]
            row_dict.update({"language": json.loads(tweet_json)["lang"]})

            if user_metadata:
                # compute metadata which would otherwise be additional burden
                user_mentions_id = []
                user_mentions_screen_name = []
                user_mentions_name = []
                for i, value in enumerate(tweet.entities['user_mentions']):
                    user_mentions_id.append(tweet.entities['user_mentions'][i]["id"])
                    user_mentions_screen_name.append(tweet.entities['user_mentions'][i]["screen_name"])
                    user_mentions_name.append(tweet.entities['user_mentions'][i]["name"])
                row_dict.update({"user_mentions_id": user_mentions_id})
                row_dict.update({"user_mentions_screen_name": user_mentions_screen_name})
                row_dict.update({"user_mentions_name": user_mentions_name})
                row_dict.update({"am_followers": json.loads(tweet_json)["user"]["followers_count"]})
                row_dict.update({"am_friends": json.loads(tweet_json)["user"]["friends_count"]})
                row_dict.update({"am_favourites": json.loads(tweet_json)["user"]["favourites_count"]})
                row_dict.update({"verified": json.loads(tweet_json)["user"]["verified"]})
                row_dict.update({"am_status": json.loads(tweet_json)["user"]["statuses_count"]})
                # append to list of rows
                row_list.append(row_dict)
            else:
                # append to list of rows
                row_list.append(row_dict)

        return pd.DataFrame(row_list)

    @staticmethod
    def data_handler_old(tweets: Any, info: List[str]) -> pd.DataFrame:
        """
        # TODO: Needs to be updated to contain all possible relevant information (and possibly be reworked)
        # TODO: Could this simply be done with pd.DataFrame(tw.Cursor(...))?
        Shortened method to extract relevant data from tw.Cursor into pd.DataFrame with info columns.
        :param tweets: tw.Cursor search, delivering filtered and downloaded tweets
        :param info: List[str], tw.Cursor results to filter from
        :return: pd.DataFrame, containing only the given/ relevant columns
        """
        # tweet. seems to be an inplace operation
        array_of_lists = np.array(
            [[tweet.user.screen_name, tweet.user.location, tweet.text]
                for tweet in tweets]).T
        # TODO: is there a way to automate data extraction? -> see above's note
        # of direct conversion to pd.DataFrame
        # Important: array_of_lists and info_list have to have the same
        # ordering, otherwise later indexing fails
        info_list = ["user", "location", "full_text"]

        # exception handling for different sized lists; not needed when array
        # is used (done by numpy then)
        if any(len(lst) != len(array_of_lists[0]) for lst in array_of_lists):
            raise ValueError("All lists need to have the same length!")

        # extract relevant information
        index = [i for i, value in enumerate(info_list) if info_list[i] in info]
        tweet_df = pd.DataFrame()
        for i, value in enumerate(index):
            tweet_df[info_list[value]] = array_of_lists[value]

        return tweet_df

    @staticmethod
    def clean_text(txt: str) -> str:
        """
        Removes URLs and special characters, as well as splitting and transforming everything to lower case.
        :param txt: str, string to be transformed
        :return: str, transformed string
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        no_url = url_pattern.sub(r'', txt)
        return re.sub('([^0-9A-Za-z \t])', '', no_url).lower().split()

    def clean_text_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts 'full_text' column of data handler DataFrame according to clean text function.
        :param df: pd.DataFrame provided by data_handler
        :return: pd.DataFrame containing no URLs and only lowercase letters and numbers
        """
        df["full_text"] = df["full_text"].map(self.clean_text)
        return df

    def get_words(self, df: pd.DataFrame, collection_words: List[str], stop_words, clean: bool = True) -> pd.DataFrame:
        """
        # TODO: implement collections_word as automatic function, extracting them from tweet element itself
        # TODO: implement stop_words as automatic function, update format
        splits tweet text into lists of words 
        :param df: pd.DataFrame with 'full_text' column
        :param collection_words: list[str], list of the word used to collect tweets
        :param stop_words: list[str], list of stopwords to remove
        :param clean: bool, if set to True (default) stop and collection words are removed
        :return: pd.DataFrame with 'full_text' transformed into list of words
        """
        df = self.clean_text_df(df)
        if clean:
            df['full_text'] = df['full_text'].apply(lambda x: [word for word in x if word not in stop_words])
            df['full_text'] = df['full_text'].apply(lambda x: [word for word in x if word not in collection_words])
            
        return df

    # TODO: Advanced request handling

    @staticmethod
    def cache():
        """
        Cache duplicate requests for rate limiting purposes
        :return:
        """
        pass

    # TODO: Data visualization, e.g. for locations with e.g. seaborn/
    # matplotlib scatter
