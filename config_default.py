"""
Contains paths to:
- working directory
- keys.csv for access keys/ tokens (containing consumer_key, consumer_secret, access_token and access_token_secret)
- save directory (for tweets in *.txt format)
- plot directory (for plots)

"""

import pathlib


class ConfigPaths(object):
    def __init__(self):
        self.work_dir = str(pathlib.Path().absolute())
        self.key_dir = '/path/to/keys.csv'
        self.save_dir = '/path/to/data/directory'
        self.plot_dir = '/path/to/plots'
