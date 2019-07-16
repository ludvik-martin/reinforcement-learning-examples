from examples.actor_critic import ActorCriticNetwork
from examples.gym_utils import NormalizedObservationWrapper, CartPoleRewardWrapper
import unittest
from unittest import TestCase
import gym
import tensorflow as tf

#%%

class ReinforceTests(TestCase):

    def test_CartPole(self):
        env = gym.make('CartPole-v1')
        writer = tf.summary.create_file_writer("/tmp/logdir/CartPole-v1")
        reinforce = ActorCriticNetwork(env, alpha=1e-3, alpha_decay=.998, min_epsilon=0, gamma=.99)
        num_episodes = 500
        for episode in range(num_episodes):
            reinforce.training_episode(debug=False)

        #reinforce.visualise_cumulative_reward(10)
        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_cumulative_reward(10)))

    def test_LunarLander(self):
        env = gym.make('LunarLander-v2')
#        writer = tf.summary.create_file_writer("/tmp/logdir/CartPole-v1")
        reinforce = ActorCriticNetwork(env, alpha=3e-3, alpha_decay=.997, min_epsilon=0, init_epsilon=0.1, gamma=.99)
        num_episodes = 1000
        for episode in range(num_episodes):
            reinforce.training_episode(debug=False, num_exploration_episodes=300)

        #reinforce.visualise_cumulative_reward(10)
        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_sum_reward(10)))

    def test_MountainCar(self):
        env = gym.make('MountainCar-v0')
        writer = tf.summary.create_file_writer("/tmp/logdir/MountainCar-v0")
        reinforce = ActorCriticNetwork(env, alpha=2e-3, alpha_decay=.998, min_epsilon=0, init_epsilon=0.3, gamma=.99)
        #        num_episodes = 800
        num_episodes = 1000
        for episode in range(num_episodes):
            reinforce.training_episode(debug=False, num_exploration_episodes=300)

        #reinforce.visualise_cumulative_reward(10)
        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_sum_reward(10)))

#%%

if __name__ == "__main__":
    unittest.main()
