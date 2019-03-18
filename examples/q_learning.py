from examples.rl_model import RLModel
from collections import defaultdict
from gym.spaces import Discrete
from gym.spaces import Box
from examples.gym_utils import DiscretizedObservationWrapper
import numpy as np

class QLearning(RLModel):
    def __init__(self, env, alpha, gamma=.99, init_epsilon = 1.0, min_epsilon = .01):
        assert isinstance(env.action_space, Discrete)
        if isinstance(env.observation_space, Box):
            env = DiscretizedObservationWrapper(env)
        else:
            assert isinstance(env.observation_space, Discrete)
        super().__init__(env, alpha, gamma, init_epsilon, min_epsilon)
        # dict with default value 0.0
        self.Q = defaultdict(float)
        self.actions = range(env.action_space.n)


    def greedy_action(self, state):
        q_values = {action: self.Q[state, action] for action in self.actions}
        q_max = max(q_values.values())
        greedy_actions = [a for a, q in q_values.items() if q == q_max]
        action = np.random.choice(greedy_actions)
        return action

    def training_episode_impl(self, episode_lenght):
        for i in range(episode_lenght):
            self.action = self.greedy_action(self.state)

            next_state, reward, done, _ = self.env.step(self.action)
            if (done):
                self.Q[self.state, self.action] = (1 - self.alpha) * self.Q[self.state, self.action] + self.alpha * (reward)
                self.state = self.env.reset()
                self.action = None
            else:
                # always use the greedy action (with the best possible Q-value)
                next_action = self.greedy_action(next_state)
                self.Q[self.state, self.action] = (1 - self.alpha) * self.Q[self.state, self.action] + self.alpha * (
                            reward + self.gamma * self.Q[next_state, next_action])
                self.state = next_state

