from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import pickle
import sys
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder

# pwd = '/home/ctilab/dti_v3/test2/dti'
pwd = './'

class IntProcessing:  # NumProcessing, NumericProcessing : data could be not only integer but also float
    def __init__(self):
        pass

    """
    save_scaling_model : save scaling model by scaling method
    
    :param
        data : train data
        save_version : path to save scaling model
        how : defined scaling method
    """
    def save_scaling_model(self, df, save_version, how=(
    'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer', 'PowerTransformer')):
        if not list(df):
            print('WARNING: THERE IS NO NUMERICAL DATA...')
            return
        data = df.copy()  # if df needs to be filtered
        # 1. choose scaling method
        if how == 'StandardScaler':
            scl_model = StandardScaler()
        elif how == 'MinMaxScaler':
            scl_model = MinMaxScaler()
        elif how == 'MaxAbsScaler':
            scl_model = MaxAbsScaler()
        elif how == 'RobustScaler':
            scl_model = RobustScaler()
        elif how == 'QuantileTransformer':
            scl_model = QuantileTransformer()
        elif how == 'PowerTransformer':
            scl_model = PowerTransformer()
        else:
            print("Scaling model [ ", how, "] is not defined")
            sys.exit()

        # 2. fit
        scl_model.fit(data)
        try:
            # 4. save model
            with open(pwd + "/model/scl_model_" + save_version + ".pickle", "wb") as f:
                pickle.dump(scl_model, f)
        except:
            raise Exception

    """
    trnsfm_scal_data : transform data by scaling model
    
    :param 
        data :
        save_version : path for scaling model
        
    :return
        scaled data
    """
    def trnsfm_scal_data(self, df, save_version):
        try:
            with open(pwd + "/model/scl_model_" + save_version + ".pickle", "rb") as f:
                scl_model = pickle.load(f)
            # 3. transform
            return scl_model.transform(df)
        except:
            raise Exception
        # except IOError as err:
        #     print(err)


class StrProcessing:
    def __init__(self):
        pass

    def tfidf_model_fit(self, df, feature, n_grams, max_features, token_pattern=r"(?u)\b\w\w+\b"):
        stop_word_list = ['bbs', 'write', 'modify', 'board', 'delete', 'id', 'contents', 'writer', 'page']
        data = df.copy()
        data.fillna(' ', inplace=True)
        col_list = list(data.columns)
        # fit_list = list(set(data[feature]))
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=n_grams, max_features=max_features,
                                           stop_words=stop_word_list, token_pattern=token_pattern)
        tfidf_vectorizer.fit(data[feature].values)
        return tfidf_vectorizer

    def save_tfidf_model_fit(self, df, feature_list, n_grams, max_features, save_version, token_pattern=r"(?u)\b\w\w+\b"):
        tfidf_model_list = []
        for feature in feature_list:
            tfidf_model = self.tfidf_model_fit(df, feature, n_grams, max_features, token_pattern)
            tfidf_model_list.append(tfidf_model)
            try:
                with open(pwd + "/model/" + str(feature) + "_tfidf_model_" + save_version + ".pickle", "wb") as f:
                    pickle.dump(tfidf_model, f)
            except:
                raise Exception
            print(feature, "model save complete **************")
        return tfidf_model_list

    def tfidf_model_trans(self, model, df, feature, batch_size):
        data = df.copy()
        temp_batch = 0
        temp_df = pd.DataFrame()
        if len(data) % batch_size == 0:
            batch_count = int(len(data) / batch_size)
        else:
            batch_count = int(len(data) / batch_size) + 1

        tf_feature = model.get_feature_names()
        for i in range(batch_count):
            if temp_batch + batch_size >= len(data):
                end_batch = len(data)
            else:
                end_batch = temp_batch + batch_size
            trans_list = list(data[feature][temp_batch: end_batch])
            tf_data = model.transform(data[feature][temp_batch: end_batch]).todense()  # instead of Pool
            temp_batch += batch_size
            tf_df = pd.DataFrame(columns=[feature + '_' + tf_name for tf_name in tf_feature], data=tf_data)
            temp_df = pd.concat([temp_df, tf_df])
        temp_df.fillna(0, inplace=True)
        temp_df.reset_index(drop=True, inplace=True)
        return temp_df, tf_feature

    def load_tfidf_model_trans(self, df, feature_list, batch_size, save_version):
        data = df.copy()
        prep_data = pd.DataFrame()
        for feature in feature_list:
            try:
                with open(pwd + "/model/" + str(feature) + "_tfidf_model_" + save_version + ".pickle", "rb") as f:
                    tfidf_model = pickle.load(f)
            except:
                raise Exception
            print(feature + " model load complete **************")

            res_df, _ = self.tfidf_model_trans(tfidf_model, data, feature, batch_size)  ### max feature
            prep_data = pd.concat([prep_data, res_df], 1)
        return prep_data

    def fit_ascii_model(self, df, feature_list, save_version):
        data = df.copy()
        data.fillna(' ', inplace=True)
        ascii_model_list = []
        asci_max_len_dict = {}
        for feature in feature_list:
            ascii_list = [[ord(ch) for ch in tmp_str] for tmp_str in data[feature].values]
            max_len = max(map(len, ascii_list))
            asci_max_len_dict.update({feature: max_len})
            # padding
            ascii_res = np.array([ascii_arr + [ord(" ")] * (max_len - len(ascii_arr)) for ascii_arr in ascii_list])
            ascii_model_list.append(ascii_res)

        try:
            # 4. save model
            with open(pwd + "/model/ascii_model_" + save_version + ".pickle", "wb") as f:
                pickle.dump(asci_max_len_dict, f)
        except:
            raise Exception

        return ascii_model_list

    def trans_ascii_model(self, df, feature_list, save_version):
        try:
            with open(pwd + "/model/ascii_model_" + save_version + ".pickle", "rb") as f:
                asci_max_len_dict = pickle.load(f)
        except:
            raise Exception
        print("ascii_model load complete **************")

        data = df.copy()
        data.fillna(' ', inplace=True)
        ascii_model_list = []
        for feature in feature_list:
            ascii_list = [[ord(ch) for ch in tmp_str] for tmp_str in data[feature].values]
            max_len = asci_max_len_dict[feature]
            # padding
            ascii_res = np.array([ascii_arr[:max_len] + [ord(" ")] * (max_len - len(ascii_arr[:max_len])) for ascii_arr in ascii_list])
            ascii_model_list.append(ascii_res)
        return ascii_model_list

    def inverse_ascii_model_to_ch(self, np_arr):
        return [[chr(i) for i in val_ar] for val_ar in np_arr]

    def inverse_ascii_model_to_str(self, np_arr):
        return [''.join([chr(i) for i in val_ar]) for val_ar in np_arr]

    # TODO
    """
    remove punctuation
    :param
        text : String Obj
    :return
        punctuation free text : String Obj
    """
    def remove_punctuation(self, text):
        punc_free = "".join([i for i in text if i not in string.punctuation])
        return punc_free

    """
    BOW : tokenization => voca => encoding
    """
    def get_tokenized_voca(self, text_arr):
        # TODO CountVectorizer with args
        # vect = CountVectorizer(min_df=5, stop_words='???').fit(text_arr)
        vect = CountVectorizer().fit(text_arr)
        return vect, vect.vocabulary_

    def get_BOW(self, vect, text_arr):
        return vect.transform(text_arr)

    def get_feature_names(self, vect):
        return vect.get_feature_names()


class CatProcessing:
    def __init__(self):
        pass

    def save_one_hot_enc_model(self, df, save_version):
        if not list(df):
            print('WARNING: THERE IS NO CATEGORICAL DATA...')
            return
        print(list(df.columns))

        data = df.copy()
        # convert numeric to string
        data = data.astype(str)
        # 1. choose cat method, sparse=False to return numpy array
        ohe_model = OneHotEncoder(sparse=False, handle_unknown='ignore')
        # 2. fit
        ohe_model.fit(data)
        try:
            # 4. save model
            with open(pwd + "/model/scl_model_" + save_version + ".pickle", "wb") as f:
                pickle.dump(ohe_model, f)
        except:
            raise Exception

    def trnsfm_one_hot_enc_data(self, df, save_version):
        try:
            with open(pwd + "/model/scl_model_" + save_version + ".pickle", "rb") as f:
                ohe_model = pickle.load(f)
            # 3. transform
            return ohe_model.transform(df)
        except:
            raise Exception

    def inverse_transform(self, df, save_version):
        try:
            with open(pwd + "/model/scl_model_" + save_version + ".pickle", "rb") as f:
                ohe_model = pickle.load(f)
            # 3. transform
            return ohe_model.inverse_transform(df)
        except:
            raise Exception


def test_BOW():
    str_proc = StrProcessing()
    punc_free = str_proc.remove_punctuation("punc[tuation... '' free text : String Obj 123")
    vec, res = str_proc.get_tokenized_voca([punc_free])
    print(res)
    res = str_proc.get_BOW(vec, [punc_free])
    print(res)
    print(str_proc.get_feature_names(vec))


if __name__ == '__main__':
    print("start >>>>>>>>>>>>>>>>>>")
    from dti_v3_data import *

    dataClass = DataCreation()

    strPrep = StrProcessing()

    print("dataClass.str_data.shape", dataClass.str_data.shape)
    import time
    start_time = time.time()
    feat_idx = 1  # 0 : agent or 1 : query
    trgt_row = 11
    save_version = 'test'
    print("feature : ", list(dataClass.str_data)[feat_idx])
    print("ori_data\n", dataClass.str_data[list(dataClass.str_data)[feat_idx]][trgt_row])
    ascii_res_list = strPrep.fit_ascii_model(dataClass.str_data, list(dataClass.str_data), save_version)
    print("ascii\n", ascii_res_list[feat_idx][trgt_row])
    ch_list = strPrep.inverse_ascii_model_to_ch(ascii_res_list[feat_idx])
    st_list = strPrep.inverse_ascii_model_to_str(ascii_res_list[feat_idx])
    print("inverse\n", ch_list[trgt_row])
    print(st_list[trgt_row], "\b:")
    print("::::::::::: %.2f seconds ::::::::::::::" % (time.time() - start_time))
