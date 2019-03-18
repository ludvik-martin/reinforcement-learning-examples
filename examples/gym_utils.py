import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

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


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        self.low = self.observation_space.low
        self.high = self.observation_space.high

    def _normalize(self, state):
        return (state - (self.high - self.low) / 2) / (self.high - self.low)

    def observation(self, observation):
        return self._normalize(observation)

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
