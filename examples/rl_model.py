from abc import ABC
from abc import abstractmethod
import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

class RLModel(ABC):
    def __init__(self, env:gym.ObservationWrapper, alpha, gamma=.99, init_epsilon = 1.0, min_epsilon = .001):
        '''
        :param env: OpenAI gym environment
        :param alpha: learning rate of the model
        :param gamma: reward discounting factor
        :param epsilon initial epsilon value for epsilon-greedy exploration/exploitation
        '''
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.init_epsilon
        self.current_episode = 0
        # current action
        self.action = None
        # current state
        self.state = None
        self.env.reset()

    def evaluate_average_cumulative_reward(self, num_episodes:int):
        '''
        :param num_episodes: number of episodes to evaluate it on
        :return:
        '''
        rewards = [self.evaluate_cumulative_reward() for i in range(num_episodes)]
        return sum(rewards) / len(rewards)

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
        return g

    def training_episode(self, num_exploration_episodes):
        # epsilon-greedy
        self.epsilon = max(self.init_epsilon * (1 - self.current_episode / num_exploration_episodes), self.min_epsilon)

        self.training_episode_impl()

    @abstractmethod
    def training_episode_impl(self):
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


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    '''
        Wrapper which converts a Box state observation (tuple of floats) into a single integer.
    '''
    def __init__(self, env, n_bins=10):
        '''
        :param env:
        :param n_bins: number of bins in each state dimension
        '''
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low
        high = self.observation_space.high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in zip(low, high)]
        self.observation_space = Discrete(n_bins ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation, self.val_bins)]
        return self._convert_to_one_number(digits)


