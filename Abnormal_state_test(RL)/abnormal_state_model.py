import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from stellargraph.mapper import PaddedGraphGenerator, PaddedGraphSequence
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import load_model
import sys
from datetime import timedelta
from RL_model import *

pwd = sys.path[0]

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

print('<'*10 + ' Enter the mode ' + '>'*10)        
mode = input()        
if mode == 'train':

    @tf.function
    class Train():
        """
        학습 모듈
        """
        def init(self):
            pass
        '''Train data Load'''
        labels_dict = {}
        train_folder = pwd + 'train_graph/'
        folders = next(os.walk(train_folder))[1]
        labels_list = []
        graphs_list = []

        for f in folders:
            if f != '.ipynb_checkpoints':
                files = next(os.walk(train_folder+f))[2]
                for file in files:
                    if f not in labels_dict.keys():
                        labels_dict[f] = len(labels_dict)

                    with open(train_folder+f+'/'+file, 'rb') as list_file:
                        graph_list = pickle.load(list_file)

                        new_labels_list = []
                        for i in range(len(graph_list)):
                            new_labels_list.append(labels_dict[f])

                        labels_list.extend(new_labels_list)
                        graphs_list.extend(graph_list)

        print(labels_dict)
        
        random_idx = np.random.permutation(len(graphs_list))
        train_X = np.array(graphs_list)[random_idx]
        train_Y = pd.get_dummies(np.array(labels_list)[random_idx])


        print('train_data : {}'.format(train_Y.sum()))

        generator = PaddedGraphGenerator(graphs=train_X)
        gen = generator.flow(
            train_X,
            targets=train_Y,
            batch_size=1
        )

        num_actions = len(list(train_Y))
        states, _ = gen.__getitem__(0)
        state_shape0 = states[0].shape[1:]
        state_shape1 = states[1].shape[1:]
        state_shape2 = states[2].shape[1:]

        env = train_environment(train_data=(train_X, train_Y), gen=gen)

        actor_model = get_actor(generator, num_actions)
        critic_model = get_critic(generator, num_actions)

        target_actor = get_actor(generator, num_actions)
        target_critic = get_critic(generator, num_actions)

        '''Making the weights equal initially'''
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        '''Learning rate for actor-critic models'''
        critic_lr = 0.0003
        actor_lr = 0.0001

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


        '''Discount factor for future rewards'''
        gamma = 0.8

        '''Used to update target networks'''
        tau = 0.001

        buffer = Buffer(num_actions, state_shape0, state_shape1, state_shape2, target_actor, target_critic, actor_model, critic_model, critic_optimizer, actor_optimizer, gamma, 10000, 64)

        print(actor_model.summary())
        print(critic_model.summary())


        '''To store reward history of each episode'''
        ep_reward_list = []
        '''To store average reward history of last few episodes'''
        avg_reward_list = []

        '''<<<<<<<<<<<<<<<<<<<<<<<<<<<< episodes EDIT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'''
        total_episodes = 400

        '''Takes about 4 min to train'''
        for ep in range(total_episodes):
            total_acc = 0
            cnt = 0

            prev_state = env.reset()
            episodic_reward = 0

            while True:
                cnt += 1

                action = policy(prev_state, actor_model, labels_dict)
                '''Recieve state and reward from environment.'''

                state, reward, done, acc = env.step(action, labels_dict)
                buffer.record((prev_state, action, reward, state))
                episodic_reward += reward
                total_acc += acc

                buffer.learn()
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)

                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward/cnt)

            if (ep+1)%10==0:
                print("Episode * {} * Avg Reward is ==> {} ACCURACY {}".format(ep+1, ep_reward_list[-1], total_acc/cnt))


        actor_model.save(pwd+'/obj/'+ 'model')
        
        if not os.path.exists(pwd+'/obj/'+ 'episodic_reward'):
            os.makedirs(pwd+'/obj/'+ 'episodic_reward')
        with open(pwd + "/obj/episodic_reward/{}".format('episodic_reward') + ".pickle", "wb") as f:
            pickle.dump(episodic_reward, f)
        
        if not os.path.exists(pwd+'/obj/'+ 'labels_dict'):
            os.makedirs(pwd+'/obj/'+ 'labels_dict')
        with open(pwd + "/obj/labels_dict/{}".format('labels_dict') + ".pickle", "wb") as f:
            pickle.dump(labels_dict, f)


        ''' Code Finish System out '''
        sys.exit()
    
else:
    class Prediction():
        """
        예측 모듈
        """
        def init(self):
            pass

        '''Prediction data load'''
        with open(pwd + "/obj/labels_dict/{}".format('labels_dict') + ".pickle", "rb") as label_file:
            labels_dict = pickle.load(label_file)
        
        with open(pwd + "/obj/episodic_reward/{}".format('episodic_reward') + ".pickle", "rb") as episodic_reward_file:
            episodic_reward = pickle.load(episodic_reward_file)
        
        pred_folder = 'pred_graph/'
        folders = next(os.walk(pred_folder))[1]
        pred_labels_list = []
        pred_graphs_list = []
        meta_df_list = []
        for f in folders:
            if f != '.ipynb_checkpoints':
                files = next(os.walk(pred_folder+f))[2]
                for file in files:
                    with open(pred_folder+f+'/'+file, 'rb') as list_file:
                        read_list = pickle.load(list_file)
                        pred_graph_list = [items[1] for items in read_list]
                        meta_df_list.append([items[0] for items in read_list])

                        new_pred_labels_list = []
                        for i in range(len(pred_graph_list)):
                            new_pred_labels_list.append(labels_dict[f])

                        pred_labels_list.extend(new_pred_labels_list)
                        pred_graphs_list.extend(pred_graph_list)

        print(labels_dict)

        random_idx = np.random.permutation(len(pred_graph_list))
        pred_X = np.array(pred_graphs_list)
        pred_Y = pd.get_dummies(np.array(pred_labels_list))

        print('prediction_data : {}'.format(pred_Y.sum()))

        pred_generator = PaddedGraphGenerator(graphs=pred_X)
        pred_gen = pred_generator.flow(
            pred_X,
            targets=pred_Y,
            batch_size=1
        )

        env = pred_environment(pred_data=(pred_X, pred_Y), gen=pred_gen)

        actor_model = load_model(pwd+'/obj/'+ 'model')

        print(actor_model.summary())
        
        total_acc = 0
        cnt = 0
        pred_dict = {}
        act_list = []
        prev_state = env.prediction_reset()
        done=False
        pred_list = []
        meta_labels_dict = {v:k for k,v in labels_dict.items()}

        while True:
            cnt += 1

            action = policy(prev_state, actor_model, labels_dict, prediction=True)
            cnt_act = np.argmax(action)

            if cnt_act in pred_dict.keys():
                pred_dict[cnt_act] += 1
            else:
                pred_dict[cnt_act] = 1
            pred_list.append(cnt_act)
            # Recieve state and reward from environment.
            state, reward, done, acc = env.prediction_step(action, labels_dict)

            if done:
                pred_dict[cnt_act] -= 1
                cnt -= 1
                break

            act_list.append(meta_labels_dict[cnt_act])

            total_acc += acc
            episodic_reward += reward


            prev_state = state
            if (cnt+1)%100==0:
                print("CNT {} * Avg ACCURACY is ==> {}".format(cnt+1, total_acc/cnt))
                
        print("ACCURACY : {}".format(total_acc/cnt))
        
        '''Counfusion_matrix'''
        matrix = pd.DataFrame()
        pred_list = pred_list[:len(pred_list)-1]
        matrix['true'] = np.argmax(pred_Y.values, 1)
        matrix['pred'] = pred_list
        
        print(confusion_matrix(matrix['true'], matrix['pred']))
        
        
        '''Make raw data & Insert Database'''
        raw_df = pd.DataFrame()
        for j in range(len(meta_df_list)):
            for k in range(len(meta_df_list[0])):
                meta_df = pd.DataFrame(data = meta_df_list[j][k], columns = ('logtime', 'src_ip', 'dst_ip', 'label'))
                '''Database frame fit'''
                meta_df['model_id'] = 400
                meta_df['end_time'] = meta_df['logtime'] + timedelta(minutes=1)
                meta_df['rule'] = '-'
                meta_df['src_ip_country_code'] = '-'
                meta_df['dst_ip_country_code'] = '-'
                meta_df['ast_ip'] = 0
                meta_df['related_ip'] = 0
                meta_df['direction_inout'] = '-'
                meta_df['direction_inout_bin'] = 0
                meta_df['related_country'] = '-'
                meta_df['result'] = 0.0
                meta_df['probs'] = '-'
                meta_df['anomaly_score'] = 0.0
                meta_df['feedback'] = '-'
                meta_df['packet'] = 0
                meta_df['if_label'] = 0
                meta_df['if_score'] = 0.0
                meta_df['lof_label'] = 0
                meta_df['lof_score'] = 0.0
                meta_df['ai_label'] = 0
                meta_df['ai_score'] = 0.0
                
                '''meta info'''
                meta_df['att_name'] = act_list[0]
                meta_df['state_idx'] = j
                
                '''model version'''
                meta_df['model_version'] = 1
                
                act_list.pop(0)
                raw_df = pd.concat([raw_df, meta_df])
                raw_df.reset_index(drop = True, inplace = True)
        
        print(raw_df[raw_df.ai_label == 'normal'][['label']].value_counts())
        
        raw_df.drop('label', axis=1, inplace=True)
        print(raw_df.shape)
        execute_ch("INSERT INTO dti.abnormal_state_result VALUES", raw_df.to_dict('records'))
        
        ''' Code Finish System out '''
        sys.exit()