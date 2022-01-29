import sys
import json
import numpy as np
import pandas as pd
from clickhouse_driver.client import Client
from clickhouse_driver.errors import ServerException, SocketTimeoutError
import pymysql
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

pwd = sys.path[0]

with open(pwd + '/conf/it_network_config.json') as f:
    config = json.loads(json.dumps(json.load(f)))
    
def make_train_data_dict(indexes, data, total_data, labels):
    train_data_dict = {}
    train_data_dict['X_train'] = data.values
    train_data_dict['Y_train'] = labels
    
    return train_data_dict


def make_pred_data_dict(data, labels):
    pred_data_dict = {}
    pred_data_dict['X_pred'] = data.values
    pred_data_dict['Y_pred'] = labels
    pred_data_dict['X_pred'] = pred_data_dict['X_pred'].reshape(pred_data_dict['X_pred'].shape[0], pred_data_dict['X_pred'].shape[1], 1)

    return pred_data_dict


def check_cs(index=0):
    cs = config['cs']
    if index >= len(cs):
        print('[clickhouse client ERROR] connect fail')
        return None
    
    '''DB(clickhouse) ip, port 정보 출력'''
    ch = cs[index]
    print(ch)
    
    try:
        client = Client(ch['host'], port=ch['port'],
                 send_receive_timeout=int(ch['timeout']),
                 settings={'max_threads': int(ch['thread'])}
                 )
        client.connection.force_connect()
        if client.connection.connected:
            return client
        else:
            return check_cs(index + 1)
    except:
        return check_cs(index + 1)

    
def execute_ch(sql, param=None, with_column_types=True):
    client = check_cs(index=0)
    print(client)
    
    if client == None:
        sys.exit(1)

    result = client.execute(sql, params=param, with_column_types=with_column_types)

    client.disconnect()
    return result
        

def get_config(model_name=None, model_id=337, TRAIN=True):
    """
    mySQL config load
    """
    conn = pymysql.connect(host = config['mysql']['host'], port = config['mysql']['port'], user = config['mysql']['user'], password = config['mysql']['password'], db = config['mysql']['db'])
    curs = conn.cursor()
    sql = 'select config from model_meta where model_id = {}'.format(model_id)
    curs.execute(sql)
    result = list(curs.fetchone())[0]

    model_config = json.loads(result)
    model_name = model_config['common']['model_name']
    conn.close()

    print('model name : [{}]\nmodel config : {}'.format(model_name, model_config))
    return model_config
            
    
def insert_result_data(result, table):
    try:
        if len(result) > 0:
            result.rename(columns = {'lgtime' : 'logtime'}, inplace = True)
            data, meta = execute_ch("""
            select * 
            from dti.it_network_AD_result""", with_column_types = True) 
            feats = [m[0] for m in meta]
            
            past_result = pd.DataFrame(data, columns = feats)        
            result = result[result.logtime.isin(past_result.logtime) == False]
            
            if len(result) > 0:
                result[list(result.select_dtypes(include = 'object'))] = result[list(result.select_dtypes(include = 'object'))].astype('str')
                execute_ch('INSERT INTO dti.{} VALUES'.format(table), result.to_dict('records'))
                return "Data insert Success"
            else:
                return "Data overlap, check predict data"
            
        else:
            return "No data, check data shape"
        
    except:
        return "Data insert Error"
            




