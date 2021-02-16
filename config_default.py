"""
Contains paths to:
- keys.csv for access keys/ tokens (containing consumer_key, consumer_secret, access_token and access_token_secret)
- working directory

"""

import pathlib


class ConfigPaths(object):
    def __init__(self):
        self.work_dir = str(pathlib.Path().absolute())
        self.key_dir = '/path/to/keys.csv'
