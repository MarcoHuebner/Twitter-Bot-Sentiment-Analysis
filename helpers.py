"""
Provides helper functions to call in (main) script

"""

import os
import numpy as np
import tweepy as tw
import pandas as pd
from typing import Any, List


class Helpers(object):
    def __init__(self):
        pass

    @staticmethod
    def init_api(path: str, row: int = 0) -> Any:
        """
        Sets tokens and returns set up API.
        :param path: str, path to token.csv, containing consumer_key, consumer_secret, access_token and
                     access_token_secret
        :param row: int, row number of key set, default 0. Added for compatibility with multiple accounts
        :return: tw.API, initialized to be waiting on rate limit
        """
        keys = pd.read_csv(path)
        auth = tw.OAuthHandler(keys.at[row, 'consumer_key'], keys.at[row, 'consumer_secret'])
        auth.set_access_token(keys.at[row, 'access_token'], keys.at[row, 'access_token_secret'])
        return tw.API(auth, wait_on_rate_limit=True)

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
        user = [tweet.user.screen_name for tweet in tweets]
        location = [tweet.user.location for tweet in tweets]
        full_text = [tweet.text for tweet in tweets]
        # is there a way to automate this? -> see above's note of direct conversion to pd.DataFrame
        # Important: array_of_lists and info_list have to have the same ordering, otherwise later indexing fails
        array_of_lists = np.array([user, location, full_text], dtype=object)
        info_list = ["user", "location", "full_text"]

        # exception handling for different sized lists
        if any(len(lst) != len(user) for lst in array_of_lists):
            raise ValueError("All lists need to have the same length!")

        # extract relevant information
        index = [i for i, value in enumerate(info_list) if info_list[i] in info]
        tweet_df = pd.DataFrame()
        for i, value in enumerate(index):
            tweet_df[info_list[value]] = array_of_lists[value]

        return tweet_df

    # TODO: Advanced request handling
    @staticmethod
    def cache():
        """
        Cache duplicate requests for rate limiting purposes
        :return:
        """
        pass

    @staticmethod
    def multi_handler():
        """
        Work balancing for multiple accounts
        :return:
        """
        pass

    # TODO: Data visualization, e.g. for locations with e.g. seaborn/ matplotlib scatter
