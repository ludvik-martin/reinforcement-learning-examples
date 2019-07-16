import tensorflow as tf
from examples.rl_model import *
from collections import defaultdict, deque
from gym.spaces import Discrete

import random
from tensorboard.plugins.hparams import summary as hparams_summary
#tf.enable_eager_execution()

class CriticModel(tf.keras.Model):
    def __init__(self, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform())
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform())
        self.dense3 = tf.keras.layers.Dense(units=1)

    def call(self, state, action):
        state = tf.concat([tf.convert_to_tensor(state), tf.one_hot(action, depth=self.action_space_size)], axis=1)
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class ReinforceModel(tf.keras.Model):
    def __init__(self, action_space_size, batch_normalization):
        super().__init__()
        self._batch_normalization = batch_normalization
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform())
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform())
        self.dense3 = tf.keras.layers.Dense(units=action_space_size)

    def call(self, state, training):
        x = self.dense1(state)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense2(x)
        if (self._batch_normalization):
            x = tf.keras.layers.BatchNormalization(axis=1)(x, training=training)
        x = self.dense3(x)
        return x

    def greedy_action(self, state):
        return tf.squeeze(tf.random.categorical(self(state, training=False), 1)).numpy()

class ActorCriticNetwork(RLModel):
    def __init__(self, env, alpha, alpha_decay, gamma=.99, init_epsilon = 1.0, min_epsilon = .01, batch_size = 32, batch_normalization = False, writer = None):
        assert isinstance(env.action_space, Discrete)
        super().__init__(env, alpha, alpha_decay, gamma, init_epsilon, min_epsilon)
        self.batch_size = batch_size
        self.model = ReinforceModel(action_space_size=env.action_space.n, batch_normalization=batch_normalization)
        self.critic_model = CriticModel(env.action_space.n)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha)
        self.baseline_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha)
        # overall step
        self.step = 0
        self.writer = writer
        self.max_len_episode = 10000
        self.reward_history = []

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

    def training_episode_impl(self, episode_lenght, debug=False):
        # Initialize the environment and get its initial state.
        self.state = self.env.reset()

        buffer = []

        episode_reward = 0
        for t in range(self.max_len_episode):
            action = self.epsilon_greedy_action(self.state)
            next_state, reward, done, info = self.env.step(
                action)  # Let the environment to execute the action, get the next state of the action, the reward of the action, whether the game is done and extra information.
            next_action = self.epsilon_greedy_action(next_state)
            buffer.append((self.state, action, reward, next_state, next_action,
                                              1 if done else 0))  # Put the (state, action, reward, next_state, next_action) quad back into the experience replay pool.
            episode_reward+=reward
            self.state = next_state

            if done:  # Exit this round and enter the next episode if the game is over.
                # print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
                break

        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action, batch_done = zip(*buffer)
        batch_state, batch_reward, batch_next_state, batch_done = \
            [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
        batch_action = np.array(batch_action, dtype=np.int32)
        batch_next_action = np.array(batch_next_action, dtype=np.int32)

        # cummulative reward for the whole episode
        tf.summary.scalar("episode_reward", episode_reward, step=self.step)

        with tf.GradientTape() as tape:
            target = self.critic_model(batch_state, batch_action)
            advantage = target - tf.stop_gradient((batch_reward + self.gamma * self.critic_model(batch_next_state, batch_next_action)))
            critic_loss = tf.reduce_mean(advantage)
            tf.summary.scalar("critic_loss", critic_loss, step=self.step)
        critic_grads = tape.gradient(critic_loss, self.critic_model.variables)
        self.baseline_optimizer.apply_gradients(grads_and_vars=zip(critic_grads, self.critic_model.variables))


        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_action,
                                                               logits=self.model(batch_state, training=True)) *
                tf.stop_gradient(target)

            )
            tf.summary.scalar("loss", loss, step=self.step)
        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))

        if debug:
            self.reward_history.append(episode_reward)
            avg_reward = np.mean(self.reward_history[-10:])
            best_reward = np.max(self.reward_history)
            print("[step:{}], best:{}, avg:{:.2f}, alpha:{:.4f}, epsilon:{}".format(self.step, best_reward, avg_reward, self.alpha, self.epsilon))
        self.step += 1
