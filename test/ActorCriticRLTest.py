from examples.actor_critic_td import ActorCriticNetworkTD
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
        reinforce = ActorCriticNetworkTD(env, alpha=1e-3, alpha_decay=.998, min_epsilon=0, gamma=.99)
        num_episodes = 5
        for episode in range(num_episodes):
            reinforce.training_episode(debug=False, episode_lenght=100)

        #reinforce.visualise_cumulative_reward(10)
        print('Average cumulative reward after episode:{} is: {}'.format(episode, reinforce.evaluate_average_cumulative_reward(10)))
#%%

if __name__ == "__main__":
    unittest.main()
