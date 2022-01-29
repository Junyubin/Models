import pickle
import sys
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder

pwd = sys.path[0]

class StrProcessing:
    def __init__(self):
        pass

    def tfidf_model_fit(self, df, feature, n_grams, max_features, token_pattern=r"(?u)\b\w\w+\b"):
        stop_word_list = ['bbs', 'write', 'modify', 'board', 'delete', 'id', 'contents', 'writer', 'page']
        data = df.copy()
        data.fillna(' ', inplace=True)
        col_list = list(data.columns)

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
                with open(pwd + "/obj/" + 'it_network_' + str(feature) + "_tfidf_model_" + save_version + ".pickle", "wb") as f:
                    pickle.dump(tfidf_model, f)
            except:
                raise Exception
            print(feature, pwd + "/obj save complete **************")
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

            temp_batch += batch_size

            flag = True
            tries = 0
            while flag and tries < 10:
                try:
                    tries += 1
                    with Pool(20) as p:
                        tf_data = p.map(model.transform, [[item] for item in trans_list])
                        p.close()
                        p.join()
                    flag = False
                except:
                    print("trial : {}".format(str(tries)))
                    pass
            tf_feature = model.get_feature_names()
            tf_df = pd.DataFrame(columns=[feature + '_' + name for name in tf_feature], data = np.concatenate([item.toarray() for item in tf_data]))
            temp_df = pd.concat([temp_df, tf_df], sort = True)

        temp_df.fillna(0, inplace=True)
        temp_df.reset_index(drop=True, inplace=True)
        return temp_df, tf_feature

    def load_tfidf_model_trans(self, df, feature_list, batch_size, save_version):
        data = df.copy()
        prep_data = pd.DataFrame()
        for feature in feature_list:
            try:
                with open(pwd + "/obj/" + 'it_network_' + str(feature) + "_tfidf_model_" + save_version + ".pickle", "rb") as f:
                    tfidf_model = pickle.load(f)
            except:
                raise Exception
            print(feature + " model load complete **************")

            res_df, _ = self.tfidf_model_trans(tfidf_model, data, feature, batch_size)
            prep_data = pd.concat([prep_data, res_df], 1)
        return prep_data

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
        data = data.astype(str)

        self.ohe_model = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.ohe_model.fit(data)
        print(self.ohe_model.categories_)
        
        try:
            with open(pwd + "/obj/it_network_one_hot_model_" + save_version + ".pickle", "wb") as f:
                pickle.dump(self.ohe_model, f)
        except:
            raise Exception

    def trnsfm_one_hot_enc_data(self, df, save_version):
        try:
            with open(pwd + "/obj/it_network_one_hot_model_" + save_version + ".pickle", "rb") as f:
                self.ohe_model = pickle.load(f)
            return self.ohe_model.transform(df)
        except:
            raise Exception

    def inverse_transform(self, df, save_version):
        try:
            with open(pwd + "/obj/it_network_one_hot_model_" + save_version + ".pickle", "rb") as f:
                self.ohe_model = pickle.load(f)
            return self.ohe_model.inverse_transform(df)
        except:
            raise Exception

    def return_categories(self, x, pred):
        categories_array = self.ohe_model.categories_
        class_dict = {}
        
        for i in range(len(categories_array[0])):
            class_dict[i] = categories_array[0][i]
        
        result_df = x.drop('label', axis = 1)
        result_df['ai_label'] = pred
        
        label_list = np.array([])
        for i in range(len(result_df)):
            label_list = np.append(label_list, class_dict[tuple(result_df.ai_label)[i]])
            
        result_df['ai_label'] = label_list
        
        return result_df
