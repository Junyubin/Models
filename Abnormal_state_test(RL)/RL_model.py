import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from stellargraph.mapper import PaddedGraphGenerator, PaddedGraphSequence
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from sklearn.metrics import accuracy_score
import tensorflow as tf
from clickhouse_driver import Client
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

class train_environment:
    def __init__(self, train_data=None, gen=None):
        self.train_X = train_data[0]
        self.train_Y = train_data[1]
        self.gen = gen
        self.data_len = len(self.train_X)
        self.current_state = None
        self.current_action = None
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.reset()
        
    def reset(self):
        rand_idx = np.random.randint(self.data_len)
        states, action = self.gen.__getitem__(rand_idx)
        state1, state2, state3 = states
        self.current_state = (state1, state2, state3)
        self.current_action = action
        return self.current_state
        
    def step(self, action, labels_dict):
        reward = -self.cce(self.current_action.reshape(len(labels_dict),), action).numpy()/20
        acc = accuracy_score([np.array(self.current_action.reshape(len(labels_dict),)).argmax(0)], [np.array(action).argmax(0)])
        done = False
        if np.random.randint(10) > 8:
            done = True
        self.reset()
        state = self.current_state
        return state, reward, done, acc

    
class pred_environment:
    def __init__(self, pred_data=None, gen=None):
        self.test_X = pred_data[0]
        self.test_Y = pred_data[1]
        self.gen = gen
        self.data_len = len(self.test_X)
        self.cnt = 0
        self.predict_state = None
        self.predict_action = None
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        
    def prediction_reset(self):
#         states, action = gen.__getitem__(self.cnt)
        states, action = self.gen.__getitem__(self.cnt)
        state1, state2, state3 = states
        self.predict_state = (state1, state2, state3)
        self.predict_action = action
        return self.predict_state
    
    def prediction_step(self, action, labels_dict):
        reward = -self.cce(self.predict_action.reshape(len(labels_dict),), action).numpy()/20
        acc = accuracy_score([np.array(self.predict_action.reshape(len(labels_dict),)).argmax(0)], [np.array(action).argmax(0)])
        done = False
        if self.cnt >= len(self.test_X):
            self.cnt = 0
            done = True
        if done:
            return self.predict_state, reward, done, acc
        self.prediction_reset()
        self.cnt += 1
        state = self.predict_state
        return state, reward, done, acc


class Buffer:
    def __init__(self, num_actions=None, state_shape0=None, state_shape1=None, state_shape2=None, target_actor=None, 
                 target_critic=None, actor_model=None, critic_model=None, critic_optimizer=None, actor_optimizer=None, gamma=None, buffer_capacity=10000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.num_actions = num_actions
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.state_shape0 = state_shape0
        self.state_shape1 = state_shape1
        self.state_shape2 = state_shape2
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer =actor_optimizer
        self.gamma = gamma

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer0 = np.zeros((self.buffer_capacity,) + self.state_shape0)
        self.state_buffer1 = np.zeros((self.buffer_capacity,) + self.state_shape1)
        self.state_buffer2 = np.zeros((self.buffer_capacity,) + self.state_shape2)
        self.state_buffer = (self.state_buffer0, self.state_buffer1, self.state_buffer2)
        
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
#         self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.next_state_buffer0 = np.zeros((self.buffer_capacity,) + self.state_shape0)
        self.next_state_buffer1 = np.zeros((self.buffer_capacity,) + self.state_shape1)
        self.next_state_buffer2 = np.zeros((self.buffer_capacity,) + self.state_shape2)
        self.next_state_buffer = (self.next_state_buffer0, self.next_state_buffer1, self.next_state_buffer2)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[0][index] = obs_tuple[0][0]
        self.state_buffer[1][index] = obs_tuple[0][1]
        self.state_buffer[2][index] = obs_tuple[0][2]
        
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[0][index] = obs_tuple[3][0]
        self.next_state_buffer[1][index] = obs_tuple[3][1]
        self.next_state_buffer[2][index] = obs_tuple[3][2]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        
        state_batch0 = tf.convert_to_tensor(self.state_buffer[0][batch_indices])
        state_batch1 = tf.convert_to_tensor(self.state_buffer[1][batch_indices])
        state_batch2 = tf.convert_to_tensor(self.state_buffer[2][batch_indices])
        state_batch = (state_batch0, state_batch1, state_batch2)
        
        
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        next_state_batch0 = tf.convert_to_tensor(self.next_state_buffer[0][batch_indices])
        next_state_batch1 = tf.convert_to_tensor(self.next_state_buffer[1][batch_indices])
        next_state_batch2 = tf.convert_to_tensor(self.next_state_buffer[2][batch_indices])
        next_state_batch = (next_state_batch0, next_state_batch1, next_state_batch2)
        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
        
        
def get_actor(generator, num_actions):
    # Initialize weights between -3e-3 and 3-e3
#     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

#     inputs = layers.Input(shape=(num_states,))
    
    
    k = 35  # the number of rows for the output tensor
    layer_sizes = [16, 16, 16, 1]

    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        k=k,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes), activation='tanh')(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    
    x_out = Conv1D(filters=32, kernel_size=5, strides=1, activation='tanh')(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=64, kernel_size=5, strides=1, activation='tanh')(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=64, activation="tanh")(x_out)
    x_out = Dense(units=64, activation="tanh")(x_out)
    outputs = layers.Dense(num_actions, activation="softmax")(x_out)

    # Our upper bound is 2.0 for Pendulum.
    model = tf.keras.Model(x_inp, outputs)
    return model


def get_critic(generator, num_actions):
    # State as input
    k = 35  # the number of rows for the output tensor
    layer_sizes = [16, 16, 16, 1]

    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        k=k,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes), activation='tanh')(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    
    x_out = Conv1D(filters=32, kernel_size=5, strides=1, activation='tanh')(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    
    x_out = Conv1D(filters=64, kernel_size=5, strides=1, activation='tanh')(x_out)

    x_out = Flatten()(x_out)
    x_out = Dense(units=64, activation="tanh")(x_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(64, activation="tanh")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([x_out, action_out])

    out = layers.Dense(64, activation="tanh")(concat)
    out = layers.Dense(64, activation="tanh")(out)
    outputs = layers.Dense(1, activation="tanh")(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([x_inp, action_input], outputs)
    return model


def policy(state, actor_model, labels_dict, prediction=True):
    e = 0.0
    if np.random.random(1)[0] < e and not prediction:
        sampled_actions = np.zeros(len(labels_dict))
        sampled_actions[np.random.randint(len(labels_dict))] += 1
    else:
        sampled_actions = tf.squeeze(actor_model(state))
    return sampled_actions

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