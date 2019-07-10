from examples.reinforce import ReinforceNetwork
from examples.gym_utils import NormalizedObservationWrapper, CartPoleRewardWrapper
import unittest
from unittest import TestCase
import gym
import tensorflow as tf

#%%

class ReinforceTests(TestCase):

    def test_CartPole(self):
        env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
        writer = tf.summary.create_file_writer("/tmp/logdir/CartPole-v1")
        reinforce = ReinforceNetwork(env, alpha=1e-3, alpha_decay=.998, min_epsilon=0, gamma=.99)
#        num_episodes = 800
        num_episodes = 10
        for episode in range(num_episodes):
            reinforce.training_episode(debug=False)

        reinforce.visualise_cumulative_reward(10)
        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_cumulative_reward(100)))

    def test_MountainCar(self):
        env = NormalizedObservationWrapper(gym.make('MountainCar-v0'))
        writer = tf.summary.create_file_writer("/tmp/logdir/MountainCar-v0")
        reinforce = ReinforceNetwork(env, alpha=1e-3, alpha_decay=.998, min_epsilon=0, gamma=.99)
        #        num_episodes = 800
        num_episodes = 100
        for episode in range(num_episodes):
            reinforce.training_episode(debug=False)

        reinforce.visualise_cumulative_reward(10)
        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_cumulative_reward(100)))

#%%

if __name__ == "__main__":
    unittest.main()
