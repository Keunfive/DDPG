import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import math
import tensorflow as tf

import Meshpkg.params as p
from Meshpkg.Env import State as get_state
from Meshpkg.Env.Step import step_class

def softmax_policy(Q_, temp = p.temp):

    temp = temp * tf.ones(Q_.shape)
    prob =  tf.math.exp(Q_/temp) / tf.reduce_sum(tf.math.exp(Q_/temp)) 
    action = tf.cast(tf.reshape(tf.random.categorical(prob, 1), [p.surf_length,]), tf.int32)
    return action

def epsilon_greedy_policy(Q_, epsilon):
    if np.random.rand() < epsilon:
        action = tf.cast(tf.convert_to_tensor(np.random.randint(p.n_actions, size = p.surf_length)), tf.int32)
        return action
    else:   
        action = tf.cast(tf.argmax(Q_, axis=1), tf.int32)
        return action

def DDPG_policy(state, actor_model, noise_input = None):
    
    state_new = tf.convert_to_tensor(get_state.get_new_state_2(np.array(state))) # (34, 19)

    if (noise_input is None):
        noise = 0
    else:
        noise = noise_input
    
    actions = np.array(actor_model(state_new)) + noise
    
    return actions