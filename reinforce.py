import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

#%%

num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 32
learning_rate = 1e-2
gamma = .99
initial_epsilon = 1.
final_epsilon = 0.01


# Q-network is used to fit Q function resemebled as the aforementioned multilayer perceptron. It inputs state and output Q-value under each action (2 dimensional under CartPole).
class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        # from unnormalized probability to probability
        return tf.nn.softmax(x)

    def predict(self, inputs):
        probabilities = self(inputs)
        # selecting action with maximal probability
        return tf.argmax(probabilities, axis=-1)


    def log_probability(self, inputs, a):
        #a = tf.convert_to_tensor(a)
        a = tf.one_hot(a, depth=2)
        probabilities = self(inputs.reshape(1,-1))
        # selecting probability of chosen action
        probability = tf.reduce_sum(probabilities * a, axis=-1)
        # selecting action with maximal probability
        return tf.math.log(probability)

#%%
def calculate_cumulative_reward(batch_reward, gamma):
    batch_reward = deque(batch_reward)
    cumulative_reward = 0
    cumulative_rewards = deque()
    for _ in range(len(batch_reward)):
        r = batch_reward.pop()
        cumulative_reward = gamma * cumulative_reward + r
        cumulative_rewards.insert(0, cumulative_reward)
    return np.array(cumulative_rewards, dtype=np.float32)

#%%


# Instantiate a game environment. The parameter is its name.
env = gym.make('CartPole-v1')
model = QNetwork()
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
replay_buffer = deque(maxlen=10000)
epsilon = initial_epsilon
for episode_id in range(num_episodes):
    # Initialize the environment and get its initial state.
    state = env.reset()
    epsilon = max(
        initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
        final_epsilon)
    for t in range(max_len_episode):
        # Render the current frame.
        #env.render()
        if random.random() < epsilon:  # Epsilon-greedy exploration strategy.
            action = env.action_space.sample()  # Choose random action with the probability of epsilon.
        else:
            action = model.predict(
                tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)).numpy()
            action = action[0]
        next_state, reward, done, info = env.step(
            action)  # Let the environment to execute the action, get the next state of the action, the reward of the action, whether the game is done and extra information.
        reward = -10. if done else reward  # Give a large negative reward if the game is over.
        replay_buffer.append((state, action, reward, next_state,
                              1 if done else 0))  # Put the (state, action, reward, next_state) quad back into the experience replay pool.
        state = next_state

        if done:  # Exit this round and enter the next episode if the game is over.
            print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
            break

    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*replay_buffer)
    batch_state, batch_reward, batch_next_state, batch_done = \
        [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
    batch_action = np.array(batch_action, dtype=np.int32)

    batch_cumulative_rewards =  calculate_cumulative_reward(batch_reward, gamma)

    for i in range(len(batch_state)):
        with tf.GradientTape() as tape:
            log_probability = model.log_probability(batch_state[i], batch_action[i])

        grads = tape.gradient(log_probability, model.variables)

        rewards = tf.convert_to_tensor(batch_cumulative_rewards)
        grads_with_reward = [-grad * rewards[i] for grad in grads]

        optimizer.apply_gradients(grads_and_vars=zip(grads_with_reward, model.variables))

#%%
import tensorflow as tf
import numpy as np

logits = tf.convert_to_tensor([[1,4,5]], dtype=np.float)

actions = tf.convert_to_tensor([1])

print(tf.reduce_sum(tf.math.log_softmax(logits) * tf.one_hot(actions, depth=3)).numpy())
print(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits).numpy())




