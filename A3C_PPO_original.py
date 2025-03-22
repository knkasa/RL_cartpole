import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys, os
import gym  
import itertools
import threading

#================================================================================================================
# Example of CartPole-v0.   Install gym (open AI gym).
# https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
# Note you need to install "pip install ale-py" and   "pip install gym[accept-rom-license]" (for license purpose)
# Notice epsilon becomes smaller as it runs.  Smaller episode means let agent decide actions more frequently.
# https://gym.openai.com/envs/#classic_control   documentation
# Example: https://qiita.com/KokiSakano/items/c8b92640b36b2ef21dbf
# Example: https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html
#================================================================================================================

# Converted loss function to PPO.  Code taken from A3C.py.  

current_dir = 'C:/Users/knkas/Desktop/PPO'
os.chdir( current_dir ) 

EPOCHS = 100000 
NUM_WORKERS = 1
epsilon = 0.05   
learning_rate = 0.0005  #(num_worker=1, 0.001 0.0008 0.0005 0.0003) (num_worker=10, 0.0005,  0.0003, 0.0001 )

class A3C:
    def __init__(self, id_name, master_network, local_network, env_name, epsilon, batch_size=64):
    
        self.id = id_name
        self.master_network = master_network
        self.local_network = local_network
        self.optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate )
        
        self.env_name = env_name
        self.input_size = 4  
        self.action_size = 2  
        self.batch_size = batch_size
        
        self.gamma = 1.0     
        self.epsilon = epsilon
        
        self.local_network.model.set_weights( weights=self.master_network.model.get_weights() )
        
    def train(self, tape, reward_list, policy_list, value_list, action_list):   
        
        action_list = tf.one_hot( action_list, self.action_size, dtype=tf.float32 ).numpy().tolist() 
        advantage = self.get_advantage( reward_list, value_list )  
        
        policy_responsible = tf.reduce_sum( tf.squeeze(policy_list)*action_list, axis=1 )
        value_loss = tf.reduce_mean( tf.square(advantage) )
        entropy = -tf.reduce_sum(  policy_list*tf.math.log( tf.clip_by_value( policy_list, 1e-10, 1 ) ) )
        policy_loss = tf.reduce_mean( tf.math.log( policy_responsible + 1e-10 )*tf.stop_gradient(advantage) )
        loss = 0.5*value_loss - policy_loss + 0.01*entropy
        
        grad = tape.gradient(target=loss, sources=self.local_network.model.trainable_variables, output_gradients=None, unconnected_gradients=tf.UnconnectedGradients.NONE)
        grad_clip, global_norm = tf.clip_by_global_norm(t_list=grad, clip_norm=5.0)
        grad_clip[0] = tf.where( tf.math.is_nan(grad_clip[0]), tf.zeros_like(grad_clip[0]), grad_clip[0] )
        rt = self.optimizer.apply_gradients( zip(grad_clip, self.master_network.model.trainable_variables) )
        
    def get_advantage(self, reward_list, value_list, gamma=0.99, standardize=True):

        try:
            tf.stack(reward_list)   # new version of Cartpole use tf.stack instead.
        except tf.python.framework.errors_impl.InvalidArgumentError as error:
            reward_list = tf.concat(reward_list, axis=0)  
            
        n = tf.shape(reward_list)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast( reward_list[::-1], dtype=tf.float32) 
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            discounted_sum = rewards[i] + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        
        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns))/(tf.math.reduce_std(returns) + 1.0e-20))
        
        advantage = returns - tf.concat( value_list, axis=0) 
        
        return advantage

    def choose_action(self, state, epsilon):   
        #return  np.random.choice( self.action_size, p=self.local_network.model(state)[0][0].numpy() )

        # Get policy from local network
        policy = self.local_network.model(state)[0][0].numpy()
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.random.choice(self.action_size, p=policy)

    def preprocess_state(self, state):
        if isinstance(state, tuple):  # Unpack if state is a tuple
            state = state[0]
        return np.reshape( state, [1, self.input_size] )
        
    def play(self):
        
        score_list = []
        for e in range(EPOCHS):
            
            env = gym.make( 'CartPole-v1' )   
            state = env.reset()  
            state = self.preprocess_state(state) 
            
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.reset()
                tape.watch( self.local_network.model.get_layer('hidden_layer').trainable_variables )
                tape.watch( self.local_network.model.get_layer('policy_layer').trainable_variables )
                tape.watch( self.local_network.model.get_layer('value_layer').trainable_variables )
            
                done = False
                i = 0      # i = step it stayed alive 
                action_list = []
                state_list = [state]
                reward_list = []
                policy_list = []
                value_list = []
                while not done: 
                    action = self.choose_action( state, self.epsilon ) 
                    
                    try:
                        next_state, reward, done, _ = env.step(action)
                    except ValueError as error:
                        next_state, reward, done, truncated, _ = env.step(action) # new version of Cartpole returns 5 elements.
                        
                    next_state = self.preprocess_state(next_state)
                    state = next_state
                    i += 1     # add 1 time step
                    
                    action_list.append(action)
                    state_list.append(state)
                    reward_list.append(reward*1.0e-2)
                    policy_list.append( self.local_network.model(state)[0] )
                    value_list.append( self.local_network.model(state)[1] )
                    
                score_list.append(i)
                mean_score = np.mean(  score_list[-20:]  )

                if self.id==0:
                    print( " ID = ", self.id, " Episode = ", e,  " Survival time = ", i,
                            ".  mean_score(last 20 trials) = ", np.round(mean_score,2), " epsilon = ", self.epsilon,
                            " Learning rate = ", learning_rate )
                    
                self.train( tape, reward_list, policy_list, value_list, action_list )  # train one episode.
                self.local_network.model.set_weights( weights=self.master_network.model.get_weights() )
                                                    
        return self.master_network
                
class Network:
    def __init__(self, master, id_name=None):
    
        if master:
            network_name = "master_model"
        else:
            network_name = "local_model" + str(id_name)
            
        activation_func = 'relu'
        num_unit = 100
        input_size = 4
        action_size = 2
    
        input_layer = tf.keras.layers.Input( shape=(input_size,), name='input_layer' )  
        hidden_layer = tf.keras.layers.Dense( num_unit, activation=activation_func, kernel_initializer=tf.random_uniform_initializer(seed=3), name='hidden_layer' )(input_layer)  
        policy_layer = tf.keras.layers.Dense( action_size, activation='softmax', kernel_initializer=tf.random_uniform_initializer(seed=3), name='policy_layer' )(hidden_layer)  
        value_layer = tf.keras.layers.Dense( 1, kernel_initializer=tf.random_uniform_initializer(seed=3), name='value_layer' )(hidden_layer)  
        self.model = tf.keras.Model( inputs=input_layer, outputs=[policy_layer, value_layer], name=network_name)
        
class Worker:
    def __init__(self, id_name, global_counter, master_network, env_name, epsilon ):
        self.id = id_name
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.master_network = master_network
        self.env_name = env_name
        self.epsilon = epsilon
        
        local_network = Network( master=False, id_name = self.id )
        self.agent = A3C( self.id, self.master_network, local_network, self.env_name, self.epsilon )
        
    def run(self, coordinator):
        self.master_network = self.agent.play() 

class lancher:
    def __init__(self):

        # create workers
        with tf.device("/cpu:0"):

            env_name = 'CartPole-v1'              
            master_network = Network( master=True)

            # learning 
            workers = []
            global_counter = itertools.count()
            for worker_id in range(NUM_WORKERS):
                worker = Worker(worker_id, global_counter, master_network, env_name, epsilon )
                workers.append(worker)

            # start multithread
            worker_threads = []
            cord = tf.train.Coordinator()
            for worker in workers:
                worker_fn = lambda: worker.run( cord )
                t = threading.Thread(target=worker_fn)
                t.start()
                worker_threads.append(t)
            
            # end multithread
            cord.join(worker_threads)
            print(" Done!! ")

def main():
    lancher()

if __name__ == "__main__":
    main()




