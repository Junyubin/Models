import os
import random
import sys
import pickle

# Copyright (c) 2017 Gangchen Hua
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

"""
E-SOINN in Python 3
Version 1.0
"""
from random import randint
from typing import overload
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from random import choice
import matplotlib.pyplot as plt
import threading
from sklearn.decomposition import PCA

pwd = os.getcwd()


class ESoinn(BaseEstimator, ClusterMixin):
    INITIAL_LABEL = -1

    def __init__(self, dim=2, max_edge_age=50, iteration_threshold=200, c1=0.001, c2=1.0):
        self.dim = dim
        self.iteration_threshold = iteration_threshold
        self.c1 = c1
        self.c2 = c2
        self.max_edge_age = max_edge_age
        self.num_signal = 0
        self._reset_state()
        self.fig = plt.figure()
        self.color = []

    def _reset_state(self):
        self.nodes = np.array([], dtype=np.float64)
        self.winning_times = []
        self.density = []
        self.N = []
        # if active
        self.won = []
        self.total_loop = 1
        self.s = []
        self.adjacent_mat = dok_matrix((0, 0), dtype=np.float64)
        self.node_labels = []
        self.labels_ = []
        self.sigs = []
        self.x_idx_list = []
        self.x_nois_idx_list = []
        self.monitor_result_list = []
        self.max_idx_monitor_result = 1

    def _set_state(self, esoinn_model):
        self.nodes = esoinn_model[0].copy()
        self.winning_times = esoinn_model[1].copy()
        self.density = esoinn_model[2].copy()
        self.N = esoinn_model[3].copy()
        self.won = esoinn_model[4].copy()
        self.total_loop = esoinn_model[5]
        self.s = esoinn_model[6].copy()
        self.adjacent_mat = esoinn_model[7].copy()
        self.node_labels = esoinn_model[8].copy()
        self.labels_ = esoinn_model[9].copy()
        self.sigs = esoinn_model[10].copy()
        self.x_idx_list = esoinn_model[11].copy()
        self.x_nois_idx_list = []

    def load_esoinn_model(self, save_version, epoch):
        try:
            with open("{}/model/esoinn_model_{}_{}.pickle".format(pwd, save_version, epoch), "rb") as f:
                esoinn_model = pickle.load(f)
                self._set_state(esoinn_model)
        except:
            raise Exception
        print("esoinn_model load and initiated **************")

    def delete_esoinn_model(self):
        import glob
        model_list = glob.glob("{}/model/esoinn_model_{}_*.pickle".format(pwd, self.save_version))
        new_model_list = [[tmp_path, int(tmp_path.split("_")[-1].replace(".pickle"))] for tmp_path in model_list]
        sorted_model_list = sorted(new_model_list, key=lambda new_model_list: new_model_list[-1])
        for tm_path in sorted_model_list[self.sava_last_N_model:]:
            try:
                os.remove(tm_path[0])
            except Exception as e:
                print(e)
                pass

    def save_esoinn_model(self):
        x_idx_list = [[] for _ in range(len(self.x_idx_list))]
        sv_data = [
            self.nodes
            , self.winning_times
            , self.density
            , self.N
            , self.won
            , self.total_loop
            , self.s
            , self.adjacent_mat
            , self.node_labels
            , self.labels_
            , self.sigs
            , x_idx_list
        ]

        try:
            os.makedirs("{}/model/".format(pwd), exist_ok=True)
            with open("{}/model/esoinn_model_{}_{}.pickle".format(pwd, self.save_version, self.total_loop - 1), "wb") as f:
                pickle.dump(sv_data, f)
        except:
            raise Exception
        print("esoinn_model_{}_{}.pickle saved\n".format(self.save_version, self.total_loop - 1))

    # get the most frequent label from each node array
    def get_label_list(self, idx_list, y_label):
        lb_arr_list = [[y_label[j][0] for j in i] for i in idx_list]
        return [max(set(lb_arr), key=lb_arr.count) for lb_arr in lb_arr_list]

    def predict(self, X_test):
        self.tmp_noise_idx_list = []
        self.tmp_non_noise_list = []

        for x in range(len(X_test)):
            self.pred_input_signal(X_test[x], x)

        return self.tmp_noise_idx_list, self.tmp_non_noise_list

    def pred_input_signal(self, signal: np.ndarray, x_idx):
        # Algorithm 3.4 (2)
        signal = self.__check_signal(signal)

        # Algorithm 3.4 (3)
        # winner has indexes of the closest node and the second closest node from new signal
        winner, dists = self.__find_nearest_nodes(2, signal)
        sim_thresholds = self.__calculate_similarity_thresholds(winner)
        # new node is noise(=attack)
        if dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]:
            self.tmp_noise_idx_list.append(x_idx)
        else:
            self.tmp_non_noise_list.append(x_idx)

    def fit(self, train_data, validation_data, epochs=100, full_shuffle_flag=True):
        """
        train data in batch manner
        :param
            train_data: list of array-like or ndarray
            validation_data: list of array-like or ndarray
            full_shuffle_flag : (True : train all data randomly)
                                (False : train like bagging)
        """
        ################# init train and validation data
        X = train_data[0]
        self.y_train = train_data[1]
        self.X_valtn = validation_data[0]
        self.Y_valtn = validation_data[1]
        ################# init train and validation data

        self._reset_state()

        if full_shuffle_flag:  # train all data randomly
            # # train with ONLY normal(=non-attack data)
            tmp_signals = [(X[x], x) for x in range(len(X)) if 'normal' == self.y_train[x][0]]
            if len(tmp_signals) > self.iteration_threshold * epochs:
                print("total iteration is too small. len(signals) :", len(tmp_signals), ", total iteration :", self.iteration_threshold * epochs)
                sys.exit()

            signals = [j for _ in range(self.iteration_threshold * epochs // len(tmp_signals) + 1) for j in tmp_signals]

            # # choose signals randomly
            random.shuffle(signals)

            for sig in signals[:self.iteration_threshold * epochs]:
                is_stop = self.train_input_signal(sig[0], sig[1])
                if is_stop:
                    break

        else:  # train like bagging
            tot_sigs = 0
            while True:
                # # choose signals randomly
                x = random.randrange(len(X))

                # # train with ONLY normal(=non-attack data)
                if 'normal' == self.y_train[x][0]:
                    is_stop = self.train_input_signal(X[x], x)
                    if is_stop:
                        break
                    tot_sigs += 1
                if tot_sigs >= self.iteration_threshold * epochs:
                    break

        return self

    def train_input_signal(self, signal: np.ndarray, x_idx):  # 1. numpy array 입력데이터 1개(이하 signal)와 그 index를 받는다.
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the
        state. So the function does batch training.
        :param signal: A new input signal
        :return:
        """
        # Algorithm 3.4 (2)
        signal = self.__check_signal(signal)  # 2. signal의 무결성 확인
        self.num_signal += 1
        self.sigs.append(signal)

        # Algorithm 3.4 (1)
        if len(self.nodes) < 2:  # 3. 최초 2개의 initial node 설정
            self.__add_node(signal, x_idx)
            return

        # Algorithm 3.4 (3)
        # winner has indexes of the closest node and the second closest node from new signal
        # 4-1. 기존 self.nodes의 노드들 중에 signal과 가장 가까운 2개 node(1등:winner[0], 2등:winner[1]) 구하고,
        # 각각의 거리(dists)구한다.
        winner, dists = self.__find_nearest_nodes(2, signal)
        # 4-2. 각 winner의 가장 먼 이웃 node거리를 threshold 값으로 구한다. (단, 이웃 node는 winner와 동일 cluster에 있다.)
        sim_thresholds = self.__calculate_similarity_thresholds(winner)
        if dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]:
            # 5-1. 가장 가까운 2개의 node(winner)와 signal의 거리가 각각의 threshold 보다 크면 새 node 생성한다.
            self.__add_node(signal, x_idx)
        else:  # 5-2. 입력 signal이 새 node가 아닌 경우 기존 node에 붙히고, density 등 update한다.
            # Algorithm 3.4 (4)
            # 5-2-1. 입력 signal과 가장 가까운 node(1등:winner[0])와 연결된 모든 엣지의 age를 1 증가
            self.__increment_edge_ages(winner[0])
            # Algorithm 3.4 (5)
            # 5-2-2. 1등(winner[0])과 2등(winner[1])의 subclass 합칠지 여부 판단
            need_add_edge, need_combine = self.__need_add_edge(winner)
            if need_add_edge:
                # Algorithm 3.4 (5)(a)
                self.__add_edge(winner)
            else:
                # Algorithm 3.4 (5)(b)
                self.__remove_edge_from_adjacent_mat(winner)
            # Algorithm 3.4 (5)(a) need combine subclasses
            if need_combine:  # 나중에 클래스 만들 때 자동으로 엮임(무시)
                self.__combine_subclass(winner)
            # Algorithm 3.4 (6) checked, maybe fixed problem N
            self.__update_density(winner[0])
            # Algorithm 3.4 (7)(8)
            self.__update_winner(winner[0], signal)
            ####################################################################################################
            # 5-2-3. 입력 signal이 새 node가 아니기 때문에, 가장 가까운 node(1등:winner[0])의 배열에 append 한다.
            # (입력 signal을 추적하기 위해 추가한 코드)
            self.x_idx_list[winner[0]].append(x_idx)
            ####################################################################################################
            # Algorithm 3.4 (8)
            self.__update_adjacent_nodes(winner[0], signal)

        # Algorithm 3.4 (9)
        self.__remove_old_edges()  # 6. hyper param 값(max_edge_age)보다 큰 엣지 제거한다.
        is_stop = False
        # Algorithm 3.4 (10)
        # 7. 입력 signal의 갯수가 iteration_threshold를 넘으면 classify 한다.
        # 이 조건문에 들어가기 전까지 계속 self.nodes에 signal을 추가한다.
        if self.num_signal % self.iteration_threshold == 0 and self.num_signal > 1:
            for i in range(len(self.won)):
                if self.won[i]:
                    self.N[i] += 1
            for i in range(len(self.won)):
                self.won[i] = False

            self.__separate_subclass()
            self.__delete_noise_nodes()  # 7-1. noise node를 self.nodes에서 제거한다.
            self.total_loop += 1
            self.__classify()  # 7-2. 남은 self.nodes를 classify 한다.
            if self.save_plt_fig:
                threading.Thread(self.plot_NN(self.total_loop - 1, "train"))
            ####################################################################################################
            is_stop = self.__fit_val()  # 7-3. 학습 중에 validation data로 모델을 평가한다.(모델 평가를 위해 추가된 코드)
            ####################################################################################################
            self.sigs.clear()
        return is_stop

    # checked
    def __combine_subclass(self, winner):
        if self.node_labels[winner[0]] == self.node_labels[winner[1]]:
            raise ValueError
        class_id = self.node_labels[winner[0]]
        node_belong_to_class_1 = self.find_all_index(self.node_labels, self.node_labels[winner[1]])
        for i in node_belong_to_class_1:
            self.node_labels[i] = class_id

    # checked
    def __remove_old_edges(self):
        for i in list(self.adjacent_mat.keys()):
            if self.adjacent_mat[i] > self.max_edge_age + 1:
                self.adjacent_mat.pop((i[0], i[1]))

    # checked
    def __remove_edge_from_adjacent_mat(self, ids):
        if (ids[0], ids[1]) in self.adjacent_mat and (ids[1], ids[0]) in self.adjacent_mat:
            self.adjacent_mat.pop((ids[0], ids[1]))
            self.adjacent_mat.pop((ids[1], ids[0]))

    # Algorithm 3.1
    def __separate_subclass(self):
        # find all local apex
        density_dict = {}
        density = list(self.density)
        for i in range(len(self.density)):
            density_dict[i] = density[i]
        class_id = 0
        while len(density_dict) > 0:
            apex = max(density_dict, key=lambda x: density_dict[x])
            ids = []
            ids.append(apex)
            self.__get_nodes_by_apex(apex, ids, density_dict)
            for i in set(ids):
                if i not in density_dict:
                    raise ValueError
                self.node_labels[i] = class_id
                density_dict.pop(i)
            class_id += 1

    def __get_nodes_by_apex(self, apex, ids, density_dict):
        new_ids = []
        pals = self.adjacent_mat[apex]
        for k in pals.keys():
            i = k[1]
            if self.density[i] <= self.density[apex] and i in density_dict and i not in ids:
                ids.append(i)
                new_ids.append(i)
        if len(new_ids) != 0:
            for i in new_ids:
                self.__get_nodes_by_apex(i, ids, density_dict)
        else:
            return

    # Algorithm 3.2, checked
    """
    :return need_add_edge, need_combine
    """
    def __need_add_edge(self, winner):
        # 1등 또는 2등이 -1 클래스에 있으면 합치지 않음
        if self.node_labels[winner[0]] == self.INITIAL_LABEL or \
                self.node_labels[winner[1]] == self.INITIAL_LABEL:
            return True, False
        # 1등과 2등이 같은 클래스에 있으면 합치지 않음 (이미 같은 클래스에 합쳐 있으므로)
        elif self.node_labels[winner[0]] == self.node_labels[winner[1]]:
            return True, False
        else:
            mean_density_0, max_density_0 = self.__mean_max_density(self.node_labels[winner[0]])
            mean_density_1, max_density_1 = self.__mean_max_density(self.node_labels[winner[1]])
            alpha_0 = self.calculate_alpha(mean_density_0, max_density_0)
            alpha_1 = self.calculate_alpha(mean_density_1, max_density_1)
            min_density = min([self.density[winner[0]], self.density[winner[1]]])
            # 서로 다른 클래스이고, min_density 보다 작으면 합침
            if alpha_0 * max_density_0 < min_density or alpha_1 * max_density_1 < min_density:  # (7),(8)
                return True, True
            else: # 서로 다른 클래스이고, min_density 보다 둘 다 크면 안 합침
                return False, False

    @staticmethod
    def calculate_alpha(mean_density, max_density):
        if max_density > 3.0 * mean_density:
            return 1.0
        elif 2.0 * mean_density < max_density <= 3.0 * mean_density:
            return 0.5
        else:
            return 0.0

    @staticmethod
    def find_all_index(ob, item):
        return [i for i, a in enumerate(ob) if a == item]

    # checked
    def __mean_max_density(self, class_id):
        node_belong_to_class = self.find_all_index(self.node_labels, class_id)
        avg_density = 0.0
        max_density = 0.0
        for i in node_belong_to_class:
            avg_density += self.density[i]
            if self.density[i] > max_density:
                max_density = self.density[i]
        avg_density /= len(node_belong_to_class)
        return avg_density, max_density

    @overload
    def __check_signal(self, signal: list) -> None:
        ...

    def __check_signal(self, signal: np.ndarray):
        """
        check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as
        self.dim. So, this method have to be called before calling functions
        that use self.dim.
        :param signal: an input signal
        """
        if isinstance(signal, list):
            signal = np.array(signal)
        if not (isinstance(signal, np.ndarray)):
            raise TypeError()
        if len(signal.shape) != 1:
            raise TypeError()
        self.dim = signal.shape[0]
        if not (hasattr(self, 'dim')):
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                raise TypeError()
        return signal

    # checked
    def __add_node(self, signal: np.ndarray, x_idx):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim))
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))  # ??
        self.N.append(1)
        self.density.append(0)
        self.s.append(0)
        self.won.append(False)
        self.node_labels.append(self.INITIAL_LABEL)
        # add index of input signal from data X, one node can represent several signals
        self.x_idx_list.append([x_idx])

    # checked
    def __find_nearest_nodes(self, num: int, signal: np.ndarray):
        n = self.nodes.shape[0]
        indexes = [0] * num
        sq_dists = [0.0] * num
        D = np.sum((self.nodes - np.array([signal] * n)) ** 2, 1)
        for i in range(num):
            indexes[i] = np.nanargmin(D)
            sq_dists[i] = D[indexes[i]]
            D[indexes[i]] = float('nan')
        return indexes, sq_dists

    # checked
    def __calculate_similarity_thresholds(self, node_indexes):
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            if len(pals) == 0:
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            else:
                pal_indexes = []
                for k in pals.keys():
                    pal_indexes.append(k[1])
                sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[i]] * len(pal_indexes))) ** 2, 1)
                sim_thresholds.append(np.max(sq_dists))
        return sim_thresholds

    # checked
    def __add_edge(self, node_indexes):
        self.__set_edge_weight(node_indexes, 1)

    # checked
    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)

    # checked
    def __set_edge_weight(self, index, weight):
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

    # checked
    def __update_winner(self, winner_index, signal):
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w) / self.winning_times[winner_index]

    # checked, maybe fixed the problem
    def __update_density(self, winner_index):
        self.winning_times[winner_index] += 1
        # if self.N[winner_index] == 0:
        #     raise ValueError
        pals = self.adjacent_mat[winner_index]
        pal_indexes = []
        for k in pals.keys():
            pal_indexes.append(k[1])
        if len(pal_indexes) != 0:
            sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[winner_index]] * len(pal_indexes))) ** 2,
                              1)
            mean_adjacent_density = np.mean(np.sqrt(sq_dists))
            p = 1.0 / ((1.0 + mean_adjacent_density) ** 2)
            self.s[winner_index] += p
            if self.N[winner_index] == 0:
                self.density[winner_index] = self.s[winner_index]
            else:
                self.density[winner_index] = self.s[winner_index] / self.N[winner_index]

        if self.s[winner_index] > 0:
            self.won[winner_index] = True

    # checked
    def __update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w) / (100 * self.winning_times[i])

    # checked
    def __delete_nodes(self, indexes):
        if not indexes:
            return
        n = len(self.winning_times)

        self.batch_noise = np.array(self.node_labels)[indexes]

        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))

        self.not_noise = np.array(self.node_labels)[remained_indexes]

        self.winning_times = [self.winning_times[i] for i in remained_indexes]
        self.N = [self.N[i] for i in remained_indexes]
        self.density = [self.density[i] for i in remained_indexes]
        self.node_labels = [self.node_labels[i] for i in remained_indexes]
        self.won = [self.won[i] for i in remained_indexes]
        self.s = [self.s[i] for i in remained_indexes]
        self.__delete_nodes_from_adjacent_mat(indexes, n, len(remained_indexes))

    # checked
    def __delete_nodes_from_adjacent_mat(self, indexes, prev_n, next_n):
        while indexes:
            next_adjacent_mat = dok_matrix((prev_n, prev_n))
            for key1, key2 in self.adjacent_mat.keys():
                if key1 == indexes[0] or key2 == indexes[0]:
                    continue
                if key1 > indexes[0]:
                    new_key1 = key1 - 1
                else:
                    new_key1 = key1
                if key2 > indexes[0]:
                    new_key2 = key2 - 1
                else:
                    new_key2 = key2
                # Because dok_matrix.__getitem__ is slow,
                # access as dictionary.
                next_adjacent_mat[new_key1, new_key2] = super(dok_matrix, self.adjacent_mat).__getitem__((key1, key2))
            self.adjacent_mat = next_adjacent_mat.copy()
            indexes = [i - 1 for i in indexes]
            indexes.pop(0)
        self.adjacent_mat.resize((next_n, next_n))

    # checked
    def __delete_noise_nodes(self):
        n = len(self.winning_times)
        noise_indexes = []
        mean_density_all = np.mean(self.density)
        for i in range(n):
            if len(self.adjacent_mat[i, :]) == 2 and self.density[i] < self.c1 * mean_density_all:
                noise_indexes.append(i)
            elif len(self.adjacent_mat[i, :]) == 1 and self.density[i] < self.c2 * mean_density_all:
                noise_indexes.append(i)
            elif len(self.adjacent_mat[i, :]) == 0:
                noise_indexes.append(i)
        self.__delete_nodes(noise_indexes)
        ######################################################################################################
        self.x_nois_idx_list = [self.x_idx_list[i] for i in range(len(self.x_idx_list)) if i in noise_indexes]
        self.x_idx_list = [self.x_idx_list[i] for i in range(len(self.x_idx_list)) if i not in noise_indexes]

    def __get_connected_node(self, index, indexes):
        new_ids = []
        pals = self.adjacent_mat[index]
        for k in pals.keys():
            i = k[1]
            if i not in indexes:
                indexes.append(i)
                new_ids.append(i)

        if len(new_ids) != 0:
            for i in new_ids:
                self.__get_connected_node(i, indexes)
        else:
            return

    # Algorithm 3.3
    def __classify(self):
        need_classified = list(range(len(self.node_labels)))
        for i in range(len(self.node_labels)):
            self.node_labels[i] = self.INITIAL_LABEL
        class_id = 0
        while len(need_classified) > 0:
            indexes = []
            index = choice(need_classified)
            #             index = need_classified[0]
            indexes.append(index)
            self.__get_connected_node(index, indexes)
            for i in indexes:
                self.node_labels[i] = class_id
                need_classified.remove(i)
            class_id += 1

    def plot_NN(self, idx, trn_tst):
        plt.figure(figsize=(10, 10))
        plt.cla()
        # for k in self.sigs:
        #     plt.plot(k[0], k[1], 'cx')
        pca = PCA(n_components=2)
        nodes_pca = pca.fit_transform(self.nodes)

        for k in self.adjacent_mat.keys():
            plt.plot(nodes_pca[k, 0], nodes_pca[k, 1], 'k', c='blue')
        # plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

        #         color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold']

        for i in range(len(nodes_pca)):
            if len(self.color) < len(nodes_pca):
                self.color.append('#%06X' % randint(0, 0xFFFFFF))

        color_dict = {}

        for i in range(len(nodes_pca)):
            if not self.node_labels[i] in color_dict:
                color_dict[self.node_labels[i]] = self.color[i]
            plt.plot(nodes_pca[i][0], nodes_pca[i][1], 'ro', c=color_dict[self.node_labels[i]])

        plt.grid(True)
        plt.show()
        os.makedirs("./figures/{}_{}_{}/".format(trn_tst, self.iteration_threshold, self.max_edge_age), exist_ok=True)
        plt.savefig('./figures/{}_{}_{}/fig_{}.jpg'.format(trn_tst, self.iteration_threshold, self.max_edge_age, str(idx)))

    def __fit_val(self):

        self.tmp_noise_idx_list = []
        self.tmp_non_noise_list = []

        for x in range(len(self.X_valtn)):
            self.pred_input_signal(self.X_valtn[x], x)

        monitor_result = self.__calculate_monitor(self.Y_valtn)

        if len(self.monitor_result_list) == 1 or self.save_best_only is False:
            self.save_esoinn_model()
            self.max_idx_monitor_result = self.total_loop - 1
        elif self.save_best_only is True and max(self.monitor_result_list[:-1]) < monitor_result:
            self.save_esoinn_model()
            self.max_idx_monitor_result = self.total_loop - 1

        if self.patience != 0 and self.patience <= len(self.monitor_result_list) - self.max_idx_monitor_result:
            print("EarlyStopping", ":" * 30)
            return True

        return False

    def __calculate_monitor(self, Y):
        tot_nois_node = len(self.tmp_noise_idx_list)
        tot_non_noise = len(self.tmp_non_noise_list)

        fp = self.get_cnt_by_label(self.tmp_noise_idx_list, Y, 'normal')
        tp = tot_nois_node - fp
        tn = self.get_cnt_by_label(self.tmp_non_noise_list, Y, 'normal')
        fn = tot_non_noise - tn

        accur = (tp + tn) / (tp + fp + fn + tn)
        preci = tp / (tp + fp)
        recal = tp / (tp + fn)
        f1_sc = 2 * recal * preci / (recal + preci)

        print("\ntp:", tp, "fn:", fn, "\nfp:", fp, "tn:", tn)
        print("epoch : {} \naccuracy : {} , precision : {} , recall : {} , f1_score : {}".format(
            "%4d" % (self.total_loop - 1), "%.4f" % accur, "%.4f" % preci, "%.4f" % recal, "%.4f" % f1_sc))

        if self.monitor in 'accuracy':
            self.monitor_result_list.append(accur)
            return accur
        elif self.monitor in 'precision':
            self.monitor_result_list.append(preci)
            return preci
        elif self.monitor in 'recall':
            self.monitor_result_list.append(recal)
            return recal
        elif self.monitor in 'f1_score':
            self.monitor_result_list.append(f1_sc)
            return f1_sc
        else:
            print("[ERROR] : set monitor option, eg : accuracy, precision, recall, f1_score")
            sys.exit()

    def check_point(self, save_version, monitor='acc', save_best_only=False,sava_last_N_model=0, save_plt_fig=False, patience=0):
        self.save_version = save_version
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.sava_last_N_model = sava_last_N_model
        self.save_plt_fig = save_plt_fig
        self.patience = patience

    def get_cnt_by_label(self, node_idx_list, label_np_arr, label_name):
        return len([i for i in node_idx_list if label_np_arr[i][0] == label_name])

    def get_evaluation(self, Y, monitor):
        self.monitor = monitor
        return self.__calculate_monitor(Y)
