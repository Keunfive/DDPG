import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow import keras
import Meshpkg.params as p

class NNmodel:

    def __init__(self):
        self.num_neighbor = p.num_neighbor
        self.n_actions = p.n_actions
        self.hidden_node = p.hidden_node
        self.state_input = (self.num_neighbor*2+1 -2)*2
        self.action_input = 2
        
        if p.act_shape == 0: # [1,2]
            self.action_nei_input = 2
        elif p.act_shape == 1: # [1, 625]
            self.action_nei_input = 25 ** 2
        elif p.act_shape == 2: # [1,20]
            self.action_nei_input = 5 * 2 * 2


    # def dueling(self):
    #     input_states = keras.layers.Input(shape =(self.state_input, ))
    #     hidden1 = keras.layers.Dense(self.hidden_node, activation = "gelu")(input_states)
    #     hidden2 = keras.layers.Dense(self.hidden_node, activation = "gelu")(hidden1)
    #     hidden3 = keras.layers.Dense(self.hidden_node/2, activation = "gelu")(hidden2)

    #     state_values = keras.layers.Dense(1)(hidden3)
    #     raw_advantages = keras.layers.Dense(self.n_actions, kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3))(hidden3)

    #     advantages = raw_advantages - keras.backend.max(raw_advantages)
    #     Q_values = state_values + advantages

    #     model = keras.Model(input_states, Q_values)
    #     model.compile(loss = self.loss_fn, optimizer = self.optimizer)

    #     return model
   
    def actor_dense(self):
        state = keras.Input(shape = (self.state_input,) )
        
        layer1 = keras.layers.Dense(self.hidden_node, activation="gelu")(state)
        layer2 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer1)
        layer3 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer2)
        
        action = keras.layers.Dense(2, 
                kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                activation = 'tanh')(layer3)
        
        model = keras.Model(inputs = state, outputs = action)
        model.compile(loss = p.actor_loss_fn, optimizer = p.actor_optimizer) # model compiling
        
        return model

    def critic_dense(self): 

        state = keras.Input(shape = (self.state_input,) )
        state_layer1 = keras.layers.Dense(self.hidden_node/4, activation="gelu")(state)
        state_layer2 = keras.layers.Dense(self.hidden_node/2, activation="gelu")(state_layer1)

        action = keras.Input(shape = (self.action_input,) )
        action_layer1 = keras.layers.Dense(self.hidden_node/2, activation="gelu")(action)

        layer0 = keras.layers.Concatenate()([state_layer2, action_layer1])
        
        layer1 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer0)
        layer2 = keras.layers.Dense(self.hidden_node, activation = 'gelu')(layer1)

        Q_value = keras.layers.Dense(1, 
                                      kernel_initializer = keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                                      activation = 'linear')(layer2)
        
        model = keras.Model(inputs = [state, action], outputs = Q_value)
        model.compile(loss = p.critic_loss_fn, optimizer = p.critic_optimizer) # model compiling

        return model
    
    