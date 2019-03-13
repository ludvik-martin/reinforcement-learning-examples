from examples.dqn import DeepQNetwork
from unittest import TestCase
import gym

#%%
class QLearningTests(TestCase):

    def test_deep_qlearning(self):
        env = gym.make('CartPole-v1')
        q_learning = DeepQNetwork(env, alpha=1e-3)
        num_episodes = 10 ** 2
        episode_lenth = 100
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)
            print("episode: {}, epsilon: {}, everage cumulative reward: {}".format(episode, q_learning.epsilon, q_learning.evaluate_average_cumulative_reward(100)))

        print('Average cumulative reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_cumulative_reward(100)))

