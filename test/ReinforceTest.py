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
        reinforce = ReinforceNetwork(env, alpha=1e-3)
        num_episodes = 10
        for episode in range(num_episodes):
            reinforce.training_episode(num_exploration_episodes=int(num_episodes * 2/3))
            print("episode: {}, epsilon: {}, everage cumulative reward: {}".format(episode, reinforce.epsilon, reinforce.evaluate_average_cumulative_reward(100)))

        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_cumulative_reward(100)))

#%%

if __name__ == "__main__":
    unittest.main()
