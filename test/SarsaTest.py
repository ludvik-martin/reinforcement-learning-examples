from examples.sarsa import Sarsa
from unittest import TestCase
import gym

#%%
class SarsaTests(TestCase):

    def test_sarsa(self):
        env = gym.make('CartPole-v1')
        sarsa = Sarsa(env, alpha=0.5)
        num_episodes = 100
        for episode in range(num_episodes):
            sarsa.training_episode(num_exploration_episodes=int(num_episodes * 2/3))
            print('Average cumulative reward after iteration:{} is: {}'.format(episode, sarsa.evaluate_average_cumulative_reward(100)))


