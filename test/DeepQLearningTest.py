from examples.dqn import DeepQNetwork
from examples.gym_utils import NormalizedObservationWrapper, CartPoleRewardWrapper
import unittest
from unittest import TestCase
import gym
import tensorflow as tf

#%%

class QLearningTests(TestCase):

    def test_deep_qlearning_cart_pole(self):
        env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
        writer = tf.summary.create_file_writer("/tmp/logdir/deep_q_cart_pole")
        q_learning = DeepQNetwork(env, alpha=1e-3, )
        num_episodes = 10
        episode_lenth = 50
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)
            print("episode: {}, epsilon: {}, everage cumulative reward: {}".format(episode, q_learning.epsilon, q_learning.evaluate_average_cumulative_reward(100)))

        print('Average cumulative reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_cumulative_reward(100)))

    def test_deep_qlearning_mountain_car(self):
        env = NormalizedObservationWrapper(gym.make('MountainCar-v0'))
        writer = tf.summary.create_file_writer("/tmp/logdir/deep_q_mountain_car")
        q_learning = DeepQNetwork(env, alpha=1e-3, writer=writer)
        num_episodes = 10
        episode_lenth = 50
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)
            print("episode: {}, epsilon: {}, everage cumulative reward: {}".format(episode, q_learning.epsilon, q_learning.evaluate_average_cumulative_reward(100)))

        print('Average cumulative reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_cumulative_reward(100)))

#%%

if __name__ == "__main__":
    unittest.main()
