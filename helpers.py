"""
Provides helper functions to call in (main) script

"""

import os
import re
import json
import nltk
import time
import warnings
import numpy as np
import pandas as pd
import tweepy as tw
import seaborn as sns

from typing import Any, List
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
    def _init_api(row: int = 0) -> Any:
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

    def _multi_init_api(self) -> List[Any]:
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
                        api_list.append(self._init_api(row=index))

                    return api_list
                else:
                    return list(self._init_api())
        else:
            raise IOError("Key directory non-existent or empty. Check the "
                          "README on renaming config_default.py and check for "
                          "correctness of given path")

    def tweet_saver(self, filename: str, search_words: List[str], lang: str, items: int) -> None:
        """
        Performs cursor_search and appends tweet._json to specified file. No "since" compatibility so far, as tweets get
        filtered by chronological movement back in time.
        :param filename: str, only specify filename, gets saved in ConfigPaths().save_dir
        :param search_words: List[str], containing the search words
        :param lang: str, specific language to search tweets in
        :param items: int, amount of items to save to file
        :return: None
        """
        # TODO: replace cursor_search with more sophisticated and encompassing method (e.g. for going back longer than
        #  weeks and/ or not only using tw.Cursor)
        start = time.time()
        tweets = self.cursor_search(search_words=search_words, lang=lang, items=items)
        # append to existing file or create new file if file is not existing
        for tweet in tweets:
            if os.path.isfile(ConfigPaths().save_dir + filename):
                with open(ConfigPaths().save_dir + filename, "a+") as file:
                    file.write("\n")
                    file.write(tweet)
            else:
                open(ConfigPaths().save_dir + filename, 'a').close()
                with open(ConfigPaths().save_dir + filename, "a") as file:
                    file.write(tweet)
        print("Searching and saving", items, "tweets took", time.time() - start, "seconds")
        return None

    @staticmethod
    def data_handler(tweets: Any, geo: bool = None, user_metadata: bool = True, from_cursor: bool = True,
                     filename: str = None) -> pd.DataFrame:
        """
        # TODO: Simplify and accelerate the function
        Creates a DataFrame containing the relevant information extracted from tweet._json.
        :param tweets: tw.Cursor search, delivering filtered and downloaded tweets
        :param geo: bool, filter for available geo data and skip if no data given
        :param user_metadata: bool, determine whether to collect user metadata like followers count or friends count as
                              well as mentioned users
        :param from_cursor: bool, depends loading behaviour (different if loading from cursor or from file)
        :param filename: optional str, filename in case the data is extracted from files
        :return: pd.DataFrame, containing only the given/ relevant columns
        """
        # set up empty list to append row_dict to (faster than DataFrame.append)
        start = time.time()
        row_list = []
        # Loading data from either cursor search or file (previously saved search)
        if from_cursor:
            tweets = tweets
        else:
            if filename is None:
                raise ValueError("You must either define a filename to read from or set from_cursor to True")
            with open(ConfigPaths().save_dir + filename) as file:
                tweets = [line.rstrip() for line in file]

        # Process data from tweets
        for tweet in tweets:
            # check for geo values if desired, skip tweet if unsuccessful
            if geo:
                if tweet.geo is None:
                    continue
            if from_cursor:
                # load data from json or through tweepy methods
                tweet_json = json.dumps(tweet._json)
            else:
                # load data from json string
                tweet_json = tweet

            loader = json.loads(tweet_json)
            row_dict = {"text": loader["text"]}
            hashtags = []
            for i, value in enumerate(loader["entities"]["hashtags"]):
                hashtags.append(loader["entities"]["hashtags"][i]["text"])
            row_dict.update({"hashtags": hashtags})
            row_dict.update({"source": loader["source"]})
            row_dict.update({"user_id": loader["user"]["id"]})
            row_dict.update({"user_screen_name": loader["user"]["screen_name"]})
            row_dict.update({"user_name": loader["user"]["name"]})
            row_dict.update({"location": loader["user"]["location"]})
            row_dict.update({"description": loader["user"]["description"]})
            row_dict.update({"protected": loader["user"]["protected"]})
            row_dict.update({"coordinates": loader["coordinates"]})
            row_dict.update({"retweet_count": loader["retweet_count"]})
            row_dict.update({"favourite_count": loader["favorite_count"]})
            # TODO: KeyError: 'possibly_sensitive'
            # possibly_sensitive = json.loads(tweet_json)["possibly_sensitive"]
            row_dict.update({"language": loader["lang"]})

            if user_metadata:
                # compute metadata which would otherwise be additional burden
                user_mentions_id = []
                user_mentions_screen_name = []
                user_mentions_name = []
                for i, value in enumerate(loader["entities"]["user_mentions"]):
                    user_mentions_id.append(loader["entities"]['user_mentions'][i]["id"])
                    user_mentions_screen_name.append(loader["entities"]['user_mentions'][i]["screen_name"])
                    user_mentions_name.append(loader["entities"]['user_mentions'][i]["name"])
                row_dict.update({"user_mentions_id": user_mentions_id})
                row_dict.update({"user_mentions_screen_name": user_mentions_screen_name})
                row_dict.update({"user_mentions_name": user_mentions_name})
                row_dict.update({"am_followers": loader["user"]["followers_count"]})
                row_dict.update({"am_friends": loader["user"]["friends_count"]})
                row_dict.update({"am_favourites": loader["user"]["favourites_count"]})
                row_dict.update({"verified": loader["user"]["verified"]})
                row_dict.update({"am_status": loader["user"]["statuses_count"]})
                # append to list of rows
                row_list.append(row_dict)
            else:
                # append to list of rows
                row_list.append(row_dict)
        print("Processing", len(row_list), "tweets took", time.time() - start, "seconds")
        return pd.DataFrame(row_list)

    @staticmethod
    def data_handler_old(tweets: Any, info: List[str]) -> pd.DataFrame:
        """
        ### Outdated ###
        Shortened method to extract relevant data from tw.Cursor into pd.DataFrame with info columns.
        :param tweets: tw.Cursor search, delivering filtered and downloaded tweets
        :param info: List[str], tw.Cursor results to filter from
        :return: pd.DataFrame, containing only the given/ relevant columns
        """
        # tweet. seems to be an inplace operation
        array_of_lists = np.array(
            [[tweet.user.screen_name, tweet.user.location, tweet.text]
                for tweet in tweets]).T

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

    def cursor_search(self, search_words: List[str], lang: str, items: int) -> List[str]:
        """
        Performs a cursor search using multiple Apps. Keeps track of order with max_id from tweet ids.
        :param search_words: List[str], search words for api.search
        :param lang: str, specific language to search tweets in
        :param items: int, amount of items to return
        :return: List[str], containing the _json properties of every tweet for processing
        """
        # initialize used api, amount of items and list of tweets
        current_api = 0
        item_counter = 0
        tweet_list = []
        api = self._multi_init_api()
        # get the first tweet and first tweet id to search consistently backwards
        for tweet_0 in tw.Cursor(api[current_api].search, q=search_words, lang=lang, include_entities=True,).items(1):
            tweet_list.append(json.dumps(tweet_0._json))
        # define cursor object to call .next() on in loop, max_id - 1 to avoid repetition of first tweet
        cursor = tw.Cursor(api[current_api].search, q=search_words, lang=lang, include_entities=True,
                           max_id=json.loads(tweet_list[0])["id"]-1).items()
        while True:
            try:
                tweet = cursor.next()
                if item_counter >= items-1:
                    break
                else:
                    tweet_list.append(json.dumps(tweet._json))
                item_counter += 1
            except tw.TweepError:
                print("Rate limit of current App reached")
                if current_api < len(api):
                    print("Switching to next App")
                    current_api += 1
                    cursor = tw.Cursor(api[current_api].search, q=search_words, lang=lang, include_entities=True,
                                       max_id=json.loads(tweet_list[-1])["id"]).items()
                else:
                    print("All App requests used, waiting 15 min before continuing")
                    current_api = 0
                    cursor = tw.Cursor(api[current_api].search, q=search_words, lang=lang, include_entities=True,
                                       max_id=json.loads(tweet_list[-1])["id"]).items()
                    time.sleep(60 * 15)
                continue
            except StopIteration:
                break

        return tweet_list

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
