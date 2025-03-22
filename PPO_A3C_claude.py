
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys, os
import gym  
import itertools
import threading

# PPO from Claude suggestions.

current_dir = 'C:/Users/knkas/Desktop/PPO'
os.chdir(current_dir) 

EPOCHS = 100000
NUM_WORKERS = 1
epsilon = 0.2  # Reduced epsilon for exploration
epsilon_decay = 0.995  # (num_workers=1 0.9995) (num_workers=10 0.995)
learning_rate = 0.0008   #(num_worker=1, 0.001 0.0008 0.0005 0.0003) (num_worker=10, 0.0005,  0.0003, 0.0001 )

class PPO:
    def __init__(self, id_name, master_network, local_network, env_name, epsilon, batch_size=64):
    
        self.id = id_name
        self.master_network = master_network
        self.local_network = local_network
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.env_name = env_name
        self.input_size = 4  
        self.action_size = 2  
        self.batch_size = batch_size
        
        self.gamma = 0.99  # Changed from 1.0 to 0.99 for better future reward discounting
        self.epsilon = epsilon
        self.clip_ratio = 0.2
        self.ppo_epochs = 4  # Number of policy update epochs
        self.gae_lambda = 0.95  # GAE lambda parameter for advantage estimation
        
        # Sync local network with master network
        self.local_network.model.set_weights(weights=self.master_network.model.get_weights())
        
    def compute_gae(self, rewards, values, next_val):
        """Compute Generalized Advantage Estimation"""
        values = values + [next_val]
        gae = 0
        returns = []
        
        # Reverse iteration for GAE calculation
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[step])
            
        return returns
        
    def train(self, states, actions, rewards, values, old_policies):
        # Convert to tensors
        states = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_policies = tf.convert_to_tensor(np.vstack(old_policies), dtype=tf.float32)
        
        # Calculate advantages and returns
        last_state = states[-1:]
        next_val = self.local_network.model(last_state)[1][0][0].numpy()
        returns = self.compute_gae(rewards, values, next_val)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = returns - tf.convert_to_tensor(values, dtype=tf.float32)
        
        # Normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        # One-hot encode actions
        actions_one_hot = tf.one_hot(actions, self.action_size, dtype=tf.float32)
        
        # Compute old action probabilities
        old_resp_outputs = tf.reduce_sum(old_policies * actions_one_hot, axis=1)
        
        # Multiple epochs of policy updating (a key PPO innovation)
        for _ in range(self.ppo_epochs):
            with tf.GradientTape() as tape:
                # Get current policy and value predictions
                new_policies, values_pred = self.local_network.model(states)
                values_pred = tf.squeeze(values_pred)
                
                # Compute new action probabilities
                new_resp_outputs = tf.reduce_sum(new_policies * actions_one_hot, axis=1)
                
                # PPO ratio and clipped objective
                ratio = new_resp_outputs / (old_resp_outputs + 1e-10)
                clip_1 = ratio * advantages
                clip_2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(clip_1, clip_2))
                
                # Value loss
                value_loss = 0.5 * tf.reduce_mean(tf.square(returns - values_pred))
                
                # Entropy bonus for exploration
                entropy = -tf.reduce_mean(tf.reduce_sum(new_policies * tf.math.log(new_policies + 1e-10), axis=1))
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Calculate and apply gradients
            gradients = tape.gradient(total_loss, self.local_network.model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)  # Reduced clip norm for stability
            
            # Remove NaN gradients
            gradients = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in gradients]
            
            # Apply gradients to master network
            self.optimizer.apply_gradients(zip(gradients, self.master_network.model.trainable_variables))
        
        # Sync local network with master network
        self.local_network.model.set_weights(self.master_network.model.get_weights())
        
        return total_loss

    def choose_action(self, state):
        #return np.random.choice( self.action_size, p=self.local_network.model(state)[0][0].numpy() )
        
        # Get policy from local network
        policy = self.local_network.model(state)[0][0].numpy()
        
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.random.choice(self.action_size, p=policy)

    def preprocess_state(self, state):
        if isinstance(state, tuple):  # Unpack if state is a tuple
            state = state[0]
        return np.reshape(state, [1, self.input_size])
        
    def play(self):
        score_list = []
        
        for e in range(EPOCHS):
            env = gym.make(self.env_name)   
            state = env.reset()  
            state = self.preprocess_state(state) 
            
            # Storage for trajectory
            states = []
            actions = []
            rewards = []
            values = []
            policies = []
            done = False
            episode_reward = 0
            
            # Collect trajectory
            while not done:
                # Store state
                states.append(state)
                
                # Get policy and value
                policy, value = self.local_network.model(state)
                policies.append(policy[0].numpy())
                values.append(value[0][0].numpy())
                
                # Select action
                action = self.choose_action(state)
                actions.append(action)
                
                # Take action
                try:
                    next_state, reward, done, _ = env.step(action)
                except ValueError as error:
                    next_state, reward, done, truncated, _ = env.step(action) # new version of Cartpole returns 5 elements.

                next_state = self.preprocess_state(next_state)
                rewards.append(reward)
                episode_reward += reward
                
                # Update state
                state = next_state
                
                # Break if episode is too long (for stability)
                if len(states) > 1000:
                    break
            
            # Track scores
            score_list.append(episode_reward)
            mean_score = np.mean(score_list[-20:]) if len(score_list) >= 20 else np.mean(score_list)
            
            # Train PPO after collecting trajectory
            loss = self.train(states, actions, rewards, values, policies)
            
            # Log progress
            if self.id == 0:
                print(f" ID = {self.id}, Episode = {e}, Steps = {len(states)}, " +
                      f"Mean(20) = {np.round(mean_score, 2)},  epsilon = {self.epsilon}" + 
                      f"Learning rate = {learning_rate}" )
                
            # Reduce exploration over time
            self.epsilon = max(0.05, self.epsilon*epsilon_decay)  # Decay epsilon, min value=0.05
                                    
        return self.master_network
                
class Network:
    def __init__(self, master, id_name=None):
    
        if master:
            network_name = "master_model"
        else:
            network_name = "local_model" + str(id_name)
            
        activation_func = 'tanh'  # Changed from relu to tanh for better gradient flow
        num_unit = 128  # Increased network capacity
        input_size = 4
        action_size = 2
    
        # Initialize weights with a smaller range for better initial performance
        kernel_init = tf.keras.initializers.glorot_normal(seed=3)
        
        input_layer = tf.keras.layers.Input(shape=(input_size,), name='input_layer')  
        hidden_layer = tf.keras.layers.Dense(num_unit, activation=activation_func, 
                                            kernel_initializer=kernel_init, name='hidden_layer')(input_layer)
        # Add a second hidden layer  
        hidden_layer2 = tf.keras.layers.Dense(64, activation=activation_func,
                                             kernel_initializer=kernel_init, name='hidden_layer2')(hidden_layer)
        policy_layer = tf.keras.layers.Dense(action_size, activation='softmax', 
                                            kernel_initializer=kernel_init, name='policy_layer')(hidden_layer2)  
        value_layer = tf.keras.layers.Dense(1, kernel_initializer=kernel_init, name='value_layer')(hidden_layer2)  
        self.model = tf.keras.Model(inputs=input_layer, outputs=[policy_layer, value_layer], name=network_name)
        
class Worker:
    def __init__(self, id_name, global_counter, master_network, env_name, epsilon):
        self.id = id_name
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.master_network = master_network
        self.env_name = env_name
        self.epsilon = epsilon
        
        local_network = Network(master=False, id_name=self.id)
        self.agent = PPO(self.id, self.master_network, local_network, self.env_name, self.epsilon)
        
    def run(self, coordinator):
        self.master_network = self.agent.play() 

class lancher:
    def __init__(self):
        # create workers
        with tf.device("/cpu:0"):
            env_name = 'CartPole-v1'              
            master_network = Network(master=True)

            # learning 
            workers = []
            global_counter = itertools.count()
            for worker_id in range(NUM_WORKERS):
                worker = Worker(worker_id, global_counter, master_network, env_name, epsilon)
                workers.append(worker)

            # start multithread
            worker_threads = []
            cord = tf.train.Coordinator()
            for worker in workers:
                worker_fn = lambda: worker.run(cord)
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