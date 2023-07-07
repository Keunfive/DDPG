import numpy as np

import Meshpkg.params as p
from Meshpkg.Env import State as get_state
from Meshpkg.Env.Step import step_class
from Meshpkg.Env.Reward import get_reward
from Meshpkg.Env.Action import get_action
from Meshpkg.Agent.policy import DDPG_policy
import tensorflow as tf

def inference_step(actor_model, critic_model, episode = None):

        Q_file = open("Inference_Q_record.txt", 'a')
        Q_file.write(f'\n \n Inference Q (Episode: {episode}) \n \n')
        reward_inf_mean = 0
        s = step_class()
        state = s.reset()
        
        for step in range(1, p.num_layer + 1):

            actions = DDPG_policy(state, actor_model)
            
            state_new = tf.convert_to_tensor(get_state.get_new_state_2(state))
            Q_ = critic_model([state_new, actions], training = True)
            
            #### txt writing ###
            Q_file.write(f'step: {step} \n\n')
            for i in range(len(actions)):
                Q_file.write(f' node: {i} / action:{actions[i]}] \n -> [{Q_[i]}]\n\n')
            #####################
            
            next_state_inf, reward_inf, dones_inf, info_inf, step_inf = s.step_func(actions, step)
            reward_inf_mean += np.average(reward_inf)
            
            if any(dones_inf) == 1:
                break

        Q_file.close()
        volume_mesh_inf = s.volume_mesh

        return volume_mesh_inf, reward_inf_mean
