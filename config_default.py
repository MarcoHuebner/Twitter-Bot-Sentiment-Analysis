"""
Contains paths to:
- keys.csv for access keys/ tokens (containing consumer_key, consumer_secret, access_token and access_token_secret)
- working directory

"""

import os

abspath = os.path.abspath(__file__)
work_dir = os.path.dirname(abspath)
key_dir = '/path/to/keys.csv'

class ConfigPaths(object):
    def __init__(self):
        self.work_dir = work_dir
        self.key_dir = key_dir
        

    
