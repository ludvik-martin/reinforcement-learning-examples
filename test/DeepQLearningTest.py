from examples.dqn import DeepQNetwork
from examples.rl_model import NormalizedObservationWrapper
from unittest import TestCase
import gym

#%%
class CartPoleRewardWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, done), done, info

    def reward(self, reward, done):
        if (done):
            # penalize this case
            return -10
        else:
            return  reward

class QLearningTests(TestCase):

    def test_deep_qlearning_cart_pole(self):
        env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
        q_learning = DeepQNetwork(env, alpha=1e-3, log_dir="/tmp/logdir/deep_q_cart_pole")
        num_episodes = 10 ** 2
        episode_lenth = 100
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)
            print("episode: {}, epsilon: {}, everage cumulative reward: {}".format(episode, q_learning.epsilon, q_learning.evaluate_average_cumulative_reward(100)))

        print('Average cumulative reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_cumulative_reward(100)))


    def test_deep_qlearning_mountain_car(self):
        env = NormalizedObservationWrapper(gym.make('MountainCar-v0'))
        q_learning = DeepQNetwork(env, alpha=1e-3, log_dir="/tmp/logdir/deep_q_mountain_car")
        num_episodes = 10 ** 2
        episode_lenth = 1000
        for episode in range(num_episodes):
            q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)
            print("episode: {}, epsilon: {}, everage cumulative reward: {}".format(episode, q_learning.epsilon, q_learning.evaluate_average_cumulative_reward(100)))

        print('Average cumulative reward after episode:{} is: {}'.format(episode, q_learning.evaluate_average_cumulative_reward(100)))
