import os
import pickle
from clickhouse_driver import Client
import json

pwd = os.getcwd()



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
## Json file load / save
def save_json(file, name='model_config.json'):
    try:
        with open(pwd+'/config/'+name, 'w') as outfile:
            json.dump(file, outfile)
        print('********** success! save_json {} **********'.format(name))
    except:
        print('********** Fail! save_json **********')

def load_json(name):
    try:
        with open(pwd+'/config/'+name) as json_file:
            model_config = json.loads(json_file.read())
        print('********** success! load_json {} **********'.format(name))
        return model_config
    except:
        print('********** Fail! load_json **********')

## pickle file load / save
def save_obj(obj, name):
    try:
        with open(pwd+'/obj/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print('********** success! save_obj: {}.pkl **********'.format(name))
    except:
        print('********** Fail! save_obj**********')

def load_obj(name):
    try:
        with open(pwd+'/obj/{}.pkl'.format(name), 'rb') as f:
            print('********** success! load_obj: {}.pkl **********'.format(name))
            return pickle.load(f)
    except:
        print('********** Fail! load_obj **********')
