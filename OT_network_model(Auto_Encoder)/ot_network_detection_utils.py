import pandas as pd
import pickle
import os

from clickhouse_driver import Client

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder

import sys

pwd = sys.path[0]

"""
데이터 불러오기
"""
## Clickhouse db check
def check_cs(index):
    try:
        client = Client('192.168.0.42', port='9001', send_receive_timeout=int(600000), settings={'max_threads': int(10)})
        client.connection.force_connect()
        if client.connection.connected:
            return client
        else:
            return check_cs(index + 1)
    except:
        return check_cs(index + 1)

## execute clickhouse db
def execute_ch(sql, param=None, with_column_types=True):
    client = check_cs(0)
    if client == None:
        sys.exit(1)
    
    result = client.execute(sql, params=param, with_column_types=with_column_types)

    client.disconnect()
    return result
