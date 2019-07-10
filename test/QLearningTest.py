from examples.q_learning import QLearning
from unittest import TestCase
import gym

#%%
class QLearningTests(TestCase):

    def test_qlearning_CartPole(self):
        env = gym.make('CartPole-v1')
        q_learning = QLearning(env, alpha=0.5)
        num_episodes = 10 ** 5
        episode_lenght = 1
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenght)

        print('Average cumulative reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_cumulative_reward(100)))


    def test_qlearning_MountainCar(self):
        env = gym.make('MountainCar-v0')
        q_learning = QLearning(env, alpha=0.5)
        num_episodes = 10 ** 5
        episode_lenght = 200
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 1/2), episode_lenght=episode_lenght)

        print('Average sum reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_sum_reward(100)))
