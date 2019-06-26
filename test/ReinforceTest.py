from examples.reinforce import ReinforceNetwork
from examples.gym_utils import NormalizedObservationWrapper, CartPoleRewardWrapper
import unittest
from unittest import TestCase
import gym
import tensorflow as tf

#%%

class ReinforceTests(TestCase):

    def test_cart_pole(self):
        env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
        writer = tf.summary.create_file_writer("/tmp/logdir/test")
        reinforce = ReinforceNetwork(env, alpha=1e-3, alpha_decay=.998, min_epsilon=0, gamma=.99)
        num_episodes = 800
        for episode in range(num_episodes):
            reinforce.training_episode(debug=True)

        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_cumulative_reward(100)))

#%%

if __name__ == "__main__":
    unittest.main()
