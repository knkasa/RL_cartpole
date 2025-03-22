import tensorflow as tf
import numpy as np
import gym

# Hyperparameters
learning_rate = 0.0003
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epochs = 10
T_horizon = 50

# Create the environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Neural Network for Actor and Critic
class PPO(tf.keras.Model):
    def __init__(self):
        super(PPO, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.actor = tf.keras.layers.Dense(action_dim, activation='softmax')
        self.critic = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

# Initialize the model and optimizer
model = PPO()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Function to compute the discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

# Function to compute the Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, gamma, lmbda):
    deltas = rewards + gamma * np.append(values[1:], 0) - values
    gae = compute_discounted_rewards(deltas, gamma * lmbda)
    return gae

# Training loop
for episode in range(1000):
    state, _ = env.reset()  # Handle the new gym API (returns state and info)
    done = False
    states, actions, rewards, values, probs = [], [], [], [], []

    for t in range(T_horizon):
        # Ensure the state is in the correct shape
        state = np.reshape(state, [1, state_dim])  # Reshape to [1, state_dim]

        # Get action probabilities and value from the model
        prob, value = model(state)
        action = np.random.choice(action_dim, p=prob.numpy()[0])

        # Handle the new gym API (step returns 5 values)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Combine terminated and truncated

        # Store data
        states.append(state[0])  # Append the state without the batch dimension
        actions.append(action)
        rewards.append(reward)
        values.append(value.numpy()[0][0])
        probs.append(prob.numpy()[0])  # Store the full probability distribution

        state = next_state
        if done:
            break

    # Compute discounted rewards and GAE
    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    gae = compute_gae(rewards, values, gamma, lmbda)
    gae = (gae - np.mean(gae)) / (np.std(gae) + 1e-8)

    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    old_probs = np.array(probs)  # Shape: [T_horizon, action_dim]
    old_values = np.array(values)

    # PPO Update
    for _ in range(K_epochs):
        with tf.GradientTape() as tape:
            new_probs_full, new_values = model(states)  # Shape: [T_horizon, action_dim]
            new_probs = tf.reduce_sum(new_probs_full * tf.one_hot(actions, action_dim), axis=1)  # Shape: [T_horizon]
            old_probs_selected = tf.reduce_sum(old_probs * tf.one_hot(actions, action_dim), axis=1)  # Shape: [T_horizon]

            ratio = new_probs / old_probs_selected
            surr1 = ratio * gae
            surr2 = tf.clip_by_value(ratio, 1 - eps_clip, 1 + eps_clip) * gae
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            critic_loss = tf.reduce_mean(tf.square(discounted_rewards - tf.squeeze(new_values)))

            # Compute entropy using the full probability distribution
            entropy = -tf.reduce_mean(tf.reduce_sum(new_probs_full * tf.math.log(new_probs_full + 1e-10), axis=1))
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {sum(rewards)}")

env.close()