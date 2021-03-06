from abc import ABC
from abc import abstractmethod
import gym
import numpy as np
import os
from gym import wrappers


class RLModel(ABC):
    def __init__(self, env:gym.ObservationWrapper, alpha, alpha_decay, gamma=.99, init_epsilon = 1.0, min_epsilon = .01):
        '''
        :param env: OpenAI gym environment
        :param alpha: learning rate of the model
        :param gamma: reward discounting factor
        :param epsilon initial epsilon value for epsilon-greedy exploration/exploitation
        '''
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.init_epsilon
        self.current_episode = 0
        # current action
        self.action = None
        # current state
        self.state = self.env.reset()

    def evaluate_average_cumulative_reward(self, num_episodes:int):
        '''
        :param num_episodes: number of episodes to evaluate it on
        :return:
        '''
        rewards = [self.evaluate_cumulative_reward() for i in range(num_episodes)]
        return sum(rewards) / len(rewards)

    def evaluate_average_sum_reward(self, num_episodes:int):
        '''
        :param num_episodes: number of episodes to evaluate it on
        :return:
        '''
        rewards = [self.evaluate_sum_reward() for i in range(num_episodes)]
        return sum(rewards) / len(rewards)


    def visualise_cumulative_reward(self, num_episodes:int, video_folder:str):
        for i in range(num_episodes):
            env = wrappers.Monitor(self.env, os.path.join(video_folder, str(i)), force=True)
            state = env.reset()
            # cumulative
            g = 0
            step = 0
            done = False
            while not done:
                action = self.greedy_action(state)
                state, reward, done, _ = env.step(action)
                g += reward * self.gamma ** step
                step += 1
            # cleaning-up
            self.state = self.env.reset()
            self.action = None

    def evaluate_cumulative_reward(self):
        state = self.env.reset()
        # cumulative
        g = 0
        step = 0
        done = False
        while not done:
            action = self.greedy_action(state)
            state, reward, done, _ = self.env.step(action)
            g += reward * self.gamma ** step
            step += 1
        # cleaning-up
        self.state = self.env.reset()
        self.action = None
        return g

    def evaluate_sum_reward(self):
        state = self.env.reset()
        # cumulative
        g = 0
        step = 0
        done = False
        while not done:
            action = self.greedy_action(state)
            state, reward, done, _ = self.env.step(action)
            g += reward
            step += 1
        # cleaning-up
        self.state = self.env.reset()
        self.action = None
        return g

    def training_episode(self, num_exploration_episodes=0, episode_lenght=None, debug=False):
        # epsilon-greedy
        if num_exploration_episodes != 0:
            self.epsilon = max(self.init_epsilon * (1 - self.current_episode / num_exploration_episodes), self.min_epsilon)
        else:
            self.epsilon = 0
        self.training_episode_impl(episode_lenght, debug)
        self.current_episode += 1
        # learning rate decay
        self.alpha *= self.alpha_decay

    @abstractmethod
    def training_episode_impl(self, episode_lenght, debug=False):
        '''
        Actual training episode of the model.
        :return:
        '''


    @abstractmethod
    def greedy_action(self, state):
        '''
        Selects best action from the model
        :param state: current state of the environment
        :return: best action
        '''
        pass

    def epsilon_greedy_action(self, state):
        if (np.random.random() < self.epsilon):
            # take random action
            action = self.env.action_space.sample()
        else:
            action = self.greedy_action(state)
        return action





