import tensorflow as tf
from examples.rl_model import *
from collections import defaultdict, deque
from gym.spaces import Discrete
import random
from tensorboard.plugins.hparams import summary as hparams_summary
#tf.enable_eager_execution()


class ReinforceModel(tf.keras.Model):
    def __init__(self, action_space_size, batch_normalization):
        super().__init__()
        self._batch_normalization = batch_normalization
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(units=action_space_size)

    def call(self, state, training):
        x = self.dense1(state)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense2(x)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense3(x)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense4(x)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense5(x)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense6(x)
        return x

    def greedy_action(self, state):
        return tf.squeeze(tf.random.categorical(self(state, training=False), 1)).numpy()

class ReinforceNetwork(RLModel):
    def __init__(self, env, alpha, gamma=.99, init_epsilon = 1.0, min_epsilon = .01, batch_size = 32, batch_normalization = False, writer = None):
        assert isinstance(env.action_space, Discrete)
        super().__init__(env, alpha, gamma, init_epsilon, min_epsilon)
        self.batch_size = batch_size
        self.model = ReinforceModel(action_space_size=env.action_space.n, batch_normalization=batch_normalization)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha)
        # overall step
        self.step = 0
        self.writer = writer
        self.max_len_episode = 10000

    def _calculate_cumulative_reward(self, batch_reward, gamma):
        batch_reward = deque(batch_reward)
        cumulative_reward = 0
        cumulative_rewards = deque()
        for _ in range(len(batch_reward)):
            r = batch_reward.pop()
            cumulative_reward = gamma * cumulative_reward + r
            cumulative_rewards.insert(0, cumulative_reward)
        return np.array(cumulative_rewards, dtype=np.float32)

    def greedy_action(self, state):
        return self.model.greedy_action(tf.convert_to_tensor([state], tf.float32))

    def training_episode_impl(self, episode_lenght):
        # Initialize the environment and get its initial state.
        self.state = self.env.reset()

        buffer = []

        for t in range(self.max_len_episode):
            action = self.epsilon_greedy_action(self.state)
            next_state, reward, done, info = self.env.step(
                action)  # Let the environment to execute the action, get the next state of the action, the reward of the action, whether the game is done and extra information.
            reward = -10. if done else reward  # Give a large negative reward if the game is over.
            buffer.append((self.state, action, reward, next_state,
                                  1 if done else 0))  # Put the (state, action, reward, next_state) quad back into the experience replay pool.
            self.state = next_state

            if done:  # Exit this round and enter the next episode if the game is over.
                # print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
                break

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*buffer)
        batch_state, batch_reward, batch_next_state, batch_done = \
            [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
        batch_action = np.array(batch_action, dtype=np.int32)

        batch_cumulative_rewards = self._calculate_cumulative_reward(batch_reward, self.gamma)

        # cummulative reward for the whole episode
        tf.summary.scalar("reward", batch_cumulative_rewards[0], step=self.step)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_action,
                                                               logits=self.model(batch_state, training=True)) *
                batch_cumulative_rewards

            )
            tf.summary.scalar("loss", loss, step=self.step)
        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))
        self.step += 1
