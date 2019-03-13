import tensorflow as tf
from examples.rl_model import *
from collections import defaultdict
from gym.spaces import Discrete
import random
tf.enable_eager_execution()


class DeepQModel(tf.keras.Model):
    def __init__(self, action_space_size):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.l2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.l3 = tf.keras.layers.Dense(action_space_size)

    def call(self, state):
        hidden = self.l1(state)
        hidden = self.l2(hidden)
        q = self.l3(hidden)
        # Q(s, a) for each action
        return q

    def greedy_action(self, state):
        q = self.call(state)
        greedy_action = tf.argmax(q, axis=-1)
        return greedy_action

    def max_q_value(self, state):
        q = self.call(state)
        max_q_value = tf.reduce_max(q, axis = -1)
        return max_q_value


class DeepQNetwork(RLModel):
    def __init__(self, env, alpha, gamma=.99, init_epsilon = 1.0, min_epsilon = .01, batch_size = 32):
        assert isinstance(env.action_space, Discrete)
        super().__init__(env, alpha, gamma, init_epsilon, min_epsilon)
        self.batch_size = batch_size
        self.model = DeepQModel(action_space_size=env.action_space.n)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=alpha)

    def greedy_action(self, state):
        return self.model.greedy_action(tf.convert_to_tensor([state], tf.float32))[0].numpy()

    def training_episode_impl(self, episode_lenght):
        self.state = self.env.reset()

        replay_buffer = []

        for i in range(episode_lenght):
            action = self.epsilon_greedy_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            replay_buffer.append((self.state, action, next_state, reward, done))
            self.state = next_state
            if i > self.batch_size:
                batch = random.sample(replay_buffer, self.batch_size)
                state_arr, action_arr, next_state_arr, reward_arr, done_arr = zip(*batch)
                state_arr, next_state_arr, reward_arr = [np.array(a, np.float32) for a in (state_arr, next_state_arr, reward_arr)]
                action_arr, done_arr = [np.array(a, np.int32) for a in (action_arr, done_arr)]

                with tf.GradientTape() as tape:
                    # TD
                    predicted_q = tf.reduce_sum(
                        self.model(tf.convert_to_tensor(state_arr)) * tf.one_hot(action_arr, depth=self.env.action_space.n), axis=-1)
                    target_q = reward + (self.gamma * self.model.max_q_value(tf.convert_to_tensor(next_state_arr))) * (1 - done_arr)
                    loss = tf.losses.mean_squared_error(target_q, predicted_q)
                grads = tape.gradient(loss, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables))
            if done:
                # not in the original DQN, but it allow to use GPU more efficiently
                self.state = self.env.reset()


