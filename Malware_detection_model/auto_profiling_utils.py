import pandas as pd
import pickle
import os

from clickhouse_driver import Client

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder

import sys

WORKING_DIRECTORY = sys.path[0]

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


"""
전처리 관련 클래스
"""

class DataPreprocessing():
    def __init__(self, version=None, mode=None, config=None):
        """
        :param version: 모델의 버전을 정의(information)
        :param mode: 현재까지는 'train' 만 제공, 향후 predict 기능 추가 예정
        TODO: predict mode 추가
        :param config: dictionary 형태로 설정값 전달 (common, train, predict, x_data_shape, y_data_shape)
        """
        self.version = version  # version 유효성 검사 필요??
        self.mode = mode
        self.save_path_model = WORKING_DIRECTORY + '/{}/{}'.format(config.get("common").get("model_path"), self.version)
        if not os.path.exists(self.save_path_model):
            os.makedirs(self.save_path_model)

            
    ## 데이터 형변환
    def type_transformation(data):
        ## 문자열 변환
        data[list(data)] = data[list(data)].astype('str')
        ## 모든 DLL의 요소 하나의 컬럼으로 압축
        temp = pd.DataFrame(columns = ['all'])
        for i in range(len(data)):    
            temp.loc[i,['all']] = str(list(set(list(data.iloc[i,:].values))))
        ## 특수문자 제거
        temp['all'] = [re.sub('[^A-Za-z0-9가-힣]', ' ', s) for s in temp['all']]
        temp.index = data.index
        return temp

    ## 키워드 데이터 벡터화
    def vec_module(self, data, col_list = 'all'):
        if self.mode == 'train':
            global vec
            vec = TfidfVectorizer().fit(data[col_list])
            df = pd.DataFrame(index = data.index, columns = vec.get_feature_names(), data = vec.transform(data[col_list]).toarray())
            self.save_model(vec, 'vectorization')
        else:
            vec = self.load_model('vectorization')
            df = pd.DataFrame(index = data.index, columns = vec.get_feature_names(), data = vec.transform(data[col_list]).toarray())
        return df

    ## 데이터 스케일링
    def scale_module(self, data):
        if self.mode == 'train':
            global scl_model
            scl_model = MinMaxScaler().fit(data)
            df = pd.DataFrame(index = data.index, columns = list(data), data = scl_model.transform(data))
            self.save_model(scl_model, 'scaler')
        else:
            scl_model = self.load_model('scaler')
            df = pd.DataFrame(index = data.index, columns = list(data), data = scl_model.transform(data))
        return df

#     ## onehotencoding
#     def encoder_module(self, data, col_list = 'hash'):
#         if self.mode == 'train':
#             global encoder
#             encoder = OneHotEncoder()
#             encoder.fit(data[col_list].values)
#             df = pd.DataFrame(index = data.index, columns = encoder.get_feature_names(), data = encoder.transform(data[[col_list]]).toarray())
#             self.save_model(encoder, 'encoder')
#         else:
#             encoder = self.load_model('encoder')
#             df = pd.DataFrame(index = data.index, columns = encoder.get_feature_names(), data = encoder.transform(data[[col_list]]).toarray())
#         return df

    ## onehotencoding
    def encoder_module(self, data):
        if self.mode == 'train':
            global encoder
            encoder = OneHotEncoder()
            encoder.fit(data.values)
            col_list = []
            for i in encoder.get_feature_names():
                col_list.append(i[i.find('_')+1:])
        
            df = pd.DataFrame(index = data.index, columns = col_list, data = encoder.transform(data.values).toarray())
            self.save_model(encoder, 'encoder')
        else:
            encoder = self.load_model('encoder')
            col_list = []
            for i in encoder.get_feature_names():
                col_list.append(i[i.find('_')+1:])
            df = pd.DataFrame(index = data.index, columns = col_list, data = encoder.transform(data.values).toarray())
        return df
        
    def save_model(self, obj, name = 'None'):
        with open('{}/{}.pickle'.format(self.save_path_model, name), "wb") as fw:
            pickle.dump(obj, fw)
            
    def load_model(self, name = 'None'):
        with open('{}/{}.pickle'.format(self.save_path_model, name), "rb") as fr:
            obj = pickle.load(fr)
        return obj