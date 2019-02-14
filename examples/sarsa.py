import gym
import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete
from collections import defaultdict

#%%
#def q_update(q_table, s, a, s_1, a_1)

# lookup table for q-values: q(s,a)
from select import epoll

# with 0.0 as default value in case item is not found in dictionary
q_table = defaultdict(float)

env = gym.make('CartPole-v1')

num_episodes = 100000
num_exploration_episodes = 80000

# initial epsilon - 100% random action
init_epsilon = 1.0
min_epsilon = .01
# Discounting factor
gamma = 0.99
# soft update param
alpha = 0.5

action_space = list(range(env.action_space.n))

def evaluate_cummulative_reward(env, num_episodes):
    state = env.reset()
    cumulative_rewards = []
    g = 0
    step = 0
    for i in range(num_episodes):
        # cumulative reward
        action = best_action(state)
        next_state, reward, done, _ = env.step(action)
        g += reward * gamma ** step
        if done:
            state = env.reset()
            step = 0
            cumulative_rewards.append(g)
            g = 0
        else:
            step += 1

    return sum(cumulative_rewards) / len(cumulative_rewards)



class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low, high)]
        self.observation_space = Discrete(n_bins ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation, self.val_bins)]
        return self._convert_to_one_number(digits)


env = DiscretizedObservationWrapper(
    env,
    n_bins=10,
    low=[-2.4, -2.0, -0.42, -3.5],
    high=[2.4, 2.0, 0.42, 3.5]
)

state = env.reset()
action = None

def best_action(state):
    q_values = {action: q_table[state, action] for action in action_space}
    q_max = max(q_values.values())
    actions = [a for a, q in q_values.items() if q == q_max]
    action = np.random.choice(actions)
    return action

def epsilon_greedy_action(state):
    if (np.random.random() < epsilon):
        # random action
        action = env.action_space.sample()
    else:
        action = best_action(state)
    return action

for i in range(num_episodes):
    # epsilon-greedy
    epsilon = max(init_epsilon * (1 - i/num_exploration_episodes), min_epsilon)

    if action == None:
        action = epsilon_greedy_action(state)

    next_state, reward, done, _ = env.step((action))
    if (done):
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward)
        state = env.reset()
        action = None
    else:
        next_action = epsilon_greedy_action(next_state)
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])
        state = next_state
        action = next_action

    # pick action
    if (i % 1000 == 0):
        cum_reward = evaluate_cummulative_reward(env, 100)
        print("i: {}, epsilon: {}, average cumulative reward: {}".format(i, epsilon, cum_reward))




#%%
