"""
Provides helper functions to call in (main) script

"""

import os
import re
import nltk
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
                auth = tw.OAuthHandler(keys.at[row, 'consumer_key'], keys.at[row, 'consumer_secret'])
                auth.set_access_token(keys.at[row, 'access_token'], keys.at[row, 'access_token_secret'])
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
            raise IOError("Key directory non-existent or empty. Check the README on renaming config_default.py and "
                          "check for correctness of given path")

    @staticmethod
    def data_handler(tweets: Any, info: List[str]) -> pd.DataFrame:
        """
        # TODO: Needs to be updated to contain all possible relevant information (and possibly be reworked)
        # TODO: Could this simply be done with pd.DataFrame(tw.Cursor(...))?
        Shortened method to extract relevant data from tw.Cursor into pd.DataFrame with info columns.
        :param tweets: tw.Cursor search, delivering filtered and downloaded tweets
        :param info: List[str], tw.Cursor results to filter from
        :return: pd.DataFrame, containing only the given/ relevant columns
        """
        # tweet. seems to be an inplace operation
        array_of_lists = np.array([[tweet.user.screen_name, tweet.user.location, tweet.text] for tweet in tweets]).T
        # TODO: is there a way to automate data extraction? -> see above's note of direct conversion to pd.DataFrame
        # Important: array_of_lists and info_list have to have the same ordering, otherwise later indexing fails
        info_list = ["user", "location", "full_text"]

        # exception handling for different sized lists; not needed when array is used (done by numpy then)
        if any(len(lst) != len(array_of_lists[0]) for lst in array_of_lists):
            raise ValueError("All lists need to have the same length!")

        # extract relevant information
        index = [i for i, value in enumerate(info_list) if info_list[i] in info]
        tweet_df = pd.DataFrame()
        for i, value in enumerate(index):
            tweet_df[info_list[value]] = array_of_lists[value]

        return tweet_df

    @staticmethod
    def remove_url(txt: str) -> str:  # TODO: add documentation
        """
        ???
        :param txt: str
        :return: str
        """
        return ' '.join(re.sub('([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', txt).split())

    # TODO: Advanced request handling
    @staticmethod
    def cache():
        """
        Cache duplicate requests for rate limiting purposes
        :return:
        """
        pass

    # TODO: Data visualization, e.g. for locations with e.g. seaborn/ matplotlib scatter
