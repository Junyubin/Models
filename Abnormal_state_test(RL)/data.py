from utils import *
from query import *
from datetime import date, timedelta
import pandas as pd
import numpy as np



class DataCreation():
    def __init__(self, model_id):
        self.model_id = model_id  
        self.today = date.today()
        self.interval = '1 minute'
        self.normal_start_time = '2021-06-08 00:00:00'
        self.attack_start_time = '2021-06-08 00:00:00'
        self.end_time = '2021-06-10 23:59:59'
        self.attack_array = ['BEACONING', 'CREDENTIAL', 'SQL_INJECTION', 'XSS']
        self.index_cols = ['lgtime', 'src_ip', 'dst_ip']
        self.data_load()
        
    
    def data_load(self):
        print('NORMAL DATA START DATETIME: ', self.normal_start_time)
        print('ATTACK DATA START DATETIME: ', self.attack_start_time)
        print('DATA END DATETIME: ', self.end_time)
        
        print('******* NORMAL DATA LOAD *******')
        try:
            result, meta = execute_ch(normal_query(self.normal_start_time, self.end_time, self.interval), with_column_types = True)
        except:
            print('ERROR: CHECK THE NORMAL DATA QUERY...')
            print(normal_query(self.normal_start_time, self.end_time, self.interval))
            return
            
            
        if not result:
            print('ERROR: NORMAL DATA NOT FOUND. PLEASE CHECK YOUR DATETIME AND FILTER SETTINGS...')
            return
            
        feats = [m[0] for m in meta]
        normal_data = pd.DataFrame(result, columns = feats)
        print('******* NORMAL DATA INFO *******')
        print(normal_data.info())
        
        print('******* ATTACK DATA LOAD *******')
        attack_data = pd.DataFrame()
        for attack in self.attack_array:
            sql = attack_query(attack, self.attack_start_time, self.end_time, self.interval)
            try:
                result, meta = execute_ch(sql, with_column_types = True)
            except:
                print('ERROR: CHECK THE ATTACK DATA QUERY...')
                print(attack_query(attack, self.attack_start_time, self.end_time, self.interval))
                return
            if not result:
                print('ERROR: ATTACK DATA {attack} NOT FOUND. PLEASE CHECK YOUR DATETIME AND FILTER SETTINGS...'.format(attack=attack))
                continue
            feats = [m[0] for m in meta]
            sql_data = pd.DataFrame(result, columns = feats)
            attack_data = pd.concat([attack_data, sql_data])
        print('******* ATTACK DATA INFO *******')
        print(attack_data.info())
        
        self.total_data = pd.concat([normal_data, attack_data]).convert_dtypes()
        
        datetime_tz_col = list(self.total_data.select_dtypes('datetimetz'))
        for col in datetime_tz_col:
            self.total_data[col] = pd.to_datetime(self.total_data[col]).dt.tz_localize(None)
            
        self.__type_check(self.total_data)
        self.__fill_null()
        
    
    def __type_check(self, df, cat_threshold=10):
        '''학습 & 예측 데이터 랜덤 샘플링'''
        split_len = int(len(df)*0.7)
        df = df.sample(frac=1).reset_index(drop=True)
        
        train_df = df[:split_len]
        pred_df = df[split_len:]
        
        self.pred_df = pred_df
        
        train_X_data = train_df.drop('label', axis = 1)
        pred_X_data = pred_df.drop('label', axis = 1)
        self.train_label_data = train_df[['label']]
        self.pred_label_data = pred_df[['label']]
        
        '''datetime , IP 데이터 분리'''
        self.train_index_data = train_X_data[self.index_cols]
        self.pred_index_data = pred_X_data[self.index_cols]
        
        '''유니크값이 2개 이상 100개 이하인 데이터 -> 카테고리 데이터'''
        train_X_data.drop(list(self.train_index_data), axis = 1, inplace=True)
        for i in list(train_X_data) :
            if train_X_data[i].nunique() >= 2 and train_X_data[i].nunique() <= cat_threshold:
                train_X_data[i] = train_X_data[i].astype('category')
                
        pred_X_data.drop(list(self.pred_index_data), axis = 1, inplace=True)
        for i in list(pred_X_data) :
            if pred_X_data[i].nunique() >= 2 and pred_X_data[i].nunique() <= cat_threshold:
                pred_X_data[i] = pred_X_data[i].astype('category')
                
        '''str, category 데이터 분리'''
        self.train_str_data = train_X_data.select_dtypes('string')
        self.train_cat_data = train_X_data.select_dtypes('category')
        
        self.pred_str_data = pred_X_data.select_dtypes('string')
        self.pred_cat_data = pred_X_data.select_dtypes('category')

        print('TRAIN INDEXES : ',list(self.train_index_data))
        print('TRAIN STRING : ',list(self.train_str_data))
        print('TRAIN CATEGORY : ',list(self.train_cat_data))
        
        print('PREDICTION INDEXES : ',list(self.pred_index_data))
        print('PREDICTION STRING : ',list(self.pred_str_data))
        print('PREDICTION CATEGORY : ',list(self.pred_cat_data))
        
        return 

    def __fill_null(self):
        self.train_str_data = self.train_str_data.fillna('-')
        for col in self.train_cat_data:
            self.train_cat_data[col] = self.train_cat_data[col].cat.add_categories("empty").fillna("empty")
            
        self.pred_str_data = self.pred_str_data.fillna('-')
        for col in self.pred_cat_data:
            self.pred_cat_data[col] = self.pred_cat_data[col].cat.add_categories("empty").fillna("empty")
            
    