from examples.dqn import DeepQNetwork
from examples.gym_utils import CartPoleRewardWrapper
import unittest
from unittest import TestCase
import gym
from hyperopt import fmin, tpe, hp
import tensorflow as tf

# Imports for the HParams plugin
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams_summary
from google.protobuf import struct_pb2

#%%

class DeepQLearningHyperoptTest(TestCase):

    def setUp(self):
        super().setUp()
        self._alpha_list = []
        self._gamma_list = []
        self._log_dir = "/tmp/logdir/deep_q_cart_pole"

    def _create_experiment_summary(self):
        alpha_list_val = struct_pb2.ListValue()
        alpha_list_val.extend(self._alpha_list)
        gamma_list_val = struct_pb2.ListValue()
        gamma_list_val.extend(self._gamma_list)
        return hparams_summary.experiment_pb(
            # The hyperparameters being changed
            hparam_infos=[
                api_pb2.HParamInfo(name='alpha',
                                   display_name='Learning rate',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=alpha_list_val),
                api_pb2.HParamInfo(name='gamma',
                                   display_name='Reward discount factor',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=gamma_list_val)
            ],
            # The metrics being tracked
            metric_infos=[
                api_pb2.MetricInfo(
                    name=api_pb2.MetricName(
                        tag='cummulative_reward'),
                    display_name='CumReward'),
            ]
        )

    def _experiment(self, args):
        alpha, gamma = args
        hparams = {'alpha':alpha, 'gamma':gamma}
        self._alpha_list.append(alpha)
        self._gamma_list.append(gamma)

        writer = tf.summary.create_file_writer(self._log_dir + "/alpha_{}_gamma_{:.3f}".format(alpha, gamma))
        with writer.as_default():
            summary_start = hparams_summary.session_start_pb(hparams=hparams)

            env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
            q_learning = DeepQNetwork(env, alpha=alpha, gamma=gamma, writer=writer)
            num_episodes = 10
            episode_lenth = 50
            for episode in range(num_episodes):
                q_learning.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)

            average_cmulative_reward = q_learning.evaluate_average_cumulative_reward(100)
            print('Average cumulative reward after episode:{} for alpha: {} and gamma: {} is: {}'.format(episode, alpha, gamma, average_cmulative_reward))
            summary_end = hparams_summary.session_end_pb(api_pb2.STATUS_SUCCESS)
            tf.summary.scalar('cummulative_reward', average_cmulative_reward, step=1, description="Average cummulative reward")
            tf.summary.import_event(tf.compat.v1.Event(summary=summary_start).SerializeToString())
            tf.summary.import_event(tf.compat.v1.Event(summary=summary_end).SerializeToString())

        # hyperopt needs negative value to minimize provided function properly based on fmin (no fmax alternative yet..)
        return -average_cmulative_reward

    def test_deep_qlearning_cart_pole(self):
        log_dir = "/tmp/logdir/deep_q_cart_pole"

        space = [hp.uniform('alpha', 1e-3, 5e-2), hp.uniform('gamma', .8, .99)]
        # minimize the values
        best = fmin(self._experiment, space, algo=tpe.suggest, max_evals=5)
        print("best hyperparameters: alpha={}, gamma={:.3f}".format(best['alpha'], best['gamma']))

        # all hyper parameters are know after all experiments
        exp_summary = self._create_experiment_summary()
        root_writer = tf.summary.create_file_writer(log_dir)
        with root_writer.as_default():
            tf.summary.import_event(tf.compat.v1.Event(summary=exp_summary).SerializeToString())

#%%

if __name__ == "__main__":
    unittest.main()
