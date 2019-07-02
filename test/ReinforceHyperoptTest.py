from examples.reinforce import ReinforceNetwork
from examples.gym_utils import CartPoleRewardWrapper, NormalizedObservationWrapper
import unittest
from unittest import TestCase
import gym
from hyperopt import fmin, tpe, hp
import tensorflow as tf
import math

# Imports for the HParams plugin
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams_summary
from google.protobuf import struct_pb2

#%%

class DeepQLearningHyperoptTest(TestCase):

    def setUp(self):
        super().setUp()
        self._alpha_list = []
        self._alpha_decay_list = []
        self._gamma_list = []
        self._init_epsilon_list = []
        self._n_exploration_episodes = []
        self._batch_norm_list= []

    def _create_experiment_summary(self):
        alpha_list_val = struct_pb2.ListValue()
        alpha_list_val.extend(self._alpha_list)
        alpha_decay_list_val = struct_pb2.ListValue()
        alpha_decay_list_val.extend(self._alpha_decay_list)
        gamma_list_val = struct_pb2.ListValue()
        gamma_list_val.extend(self._gamma_list)
        init_epsilon_list_val = struct_pb2.ListValue()
        init_epsilon_list_val.extend(self._init_epsilon_list)
        n_exploration_episodes_val = struct_pb2.ListValue()
        n_exploration_episodes_val.extend(self._n_exploration_episodes)
        batch_norm_list_val = struct_pb2.ListValue()
        batch_norm_list_val.extend(self._batch_norm_list)
        return hparams_summary.experiment_pb(
            # The hyperparameters being changed
            hparam_infos=[
                api_pb2.HParamInfo(name='alpha',
                                   display_name='Learning rate',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=alpha_list_val),
                api_pb2.HParamInfo(name='alpha_decay',
                                   display_name='Learning rate decay',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=alpha_decay_list_val),
                api_pb2.HParamInfo(name='gamma',
                                   display_name='Reward discount factor',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=gamma_list_val),
                api_pb2.HParamInfo(name='init_epsilon',
                                   display_name='Initial exploration',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=init_epsilon_list_val),
                api_pb2.HParamInfo(name='n_exploration_episodes',
                                   display_name='Initial exploration',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=n_exploration_episodes_val),
                api_pb2.HParamInfo(name='batch_norm',
                                   display_name='Batch normalization',
                                   type=api_pb2.DATA_TYPE_BOOL,
                                   domain_discrete=batch_norm_list_val)
            ],
            # The metrics being tracked
            metric_infos=[
                api_pb2.MetricInfo(
                    name=api_pb2.MetricName(
                        tag='sum_reward'),
                    display_name='SumReward'),
            ]
        )

    def _experiment(self, env):
        def _experiment_impl(self, args):
            alpha, alpha_decay, gamma, init_epsilon, n_exploration_episodes, batch_norm = args
            hparams = {'alpha':alpha, 'alpha_decay':alpha_decay, 'gamma':gamma, 'init_epsilon':init_epsilon, 'n_exploration_episodes':n_exploration_episodes, 'batch_norm':batch_norm}
            self._alpha_list.append(alpha)
            self._alpha_decay_list.append(alpha_decay)
            self._gamma_list.append(gamma)
            self._init_epsilon_list.append(init_epsilon)
            self._n_exploration_episodes.append(n_exploration_episodes)
            self._batch_norm_list.append(batch_norm)

            writer = tf.summary.create_file_writer(self._log_dir + "/alpha_{}_alpha_decay_{}_gamma_{:.3f}_init_eps{}_n_explor_{}_bn_{}"
                                                   .format(alpha, alpha_decay, gamma, init_epsilon, n_exploration_episodes, batch_norm))
            with writer.as_default():
                summary_start = hparams_summary.session_start_pb(hparams=hparams)

                reinforce = ReinforceNetwork(env, alpha=alpha, alpha_decay=alpha_decay, gamma=gamma, init_epsilon=init_epsilon, min_epsilon=0.0,
                                             batch_normalization=batch_norm, writer=writer)
                num_episodes = 1000
                for episode in range(num_episodes):
                    reinforce.training_episode(num_exploration_episodes=n_exploration_episodes, debug=False)

                average_sum_reward = reinforce.evaluate_average_sum_reward(5)
                print('Average sum reward after episode:{} for alpha: {}, alpha_decay: {}, gamma: {}, init_eps: {}, n_exploration_episodes: {}, batch_norm: {}, reward: {}'.
                      format(episode, alpha, alpha_decay, gamma, init_epsilon, n_exploration_episodes, batch_norm, average_sum_reward))
                summary_end = hparams_summary.session_end_pb(api_pb2.STATUS_SUCCESS)
                tf.summary.scalar('sum_reward', average_sum_reward, step=1, description="Average sum reward")
                tf.summary.import_event(tf.compat.v1.Event(summary=summary_start).SerializeToString())
                tf.summary.import_event(tf.compat.v1.Event(summary=summary_end).SerializeToString())

            # hyperopt needs negative value to minimize provided function properly based on fmin (no fmax alternative yet..)
            return -average_sum_reward

        return _experiment_impl

    def test_reinforce_cart_pole(self):
        log_dir = "/tmp/logdir/CartPole-v1"

        space = [hp.loguniform('alpha', math.log(1e-4), math.log(1e-2)), hp.uniform('alpha_decay', .995, .999),
                 hp.uniform('gamma', .99, .998), hp.uniform('init_epsilon', 0.0, 0.5),
                 hp.uniform('n_exploration_episodes', 0, 500), hp.choice('batch_norm', [True, False])
                ]
        # minimize the values
        env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
        best = fmin(self._experiment(env), space, algo=tpe.suggest, max_evals=50)
        print("best hyperparameters: alpha={}, alpha_decay={}, gamma={:.3f}, init_epsilon:{}, batch_norm:{}".format(best['alpha'],
            best['alpha_decay'], best['gamma'], best['init_epsilon'], best['batch_norm']))

        # all hyper parameters are know after all experiments
        exp_summary = self._create_experiment_summary()
        root_writer = tf.summary.create_file_writer(log_dir)
        with root_writer.as_default():
            tf.summary.import_event(tf.compat.v1.Event(summary=exp_summary).SerializeToString())


    def test_reinforce_mountain_car(self):
        log_dir = "/tmp/logdir/MountainCar-v0"

        space = [hp.loguniform('alpha', math.log(1e-4), math.log(1e-2)), hp.uniform('alpha_decay', .995, .999),
                 hp.uniform('gamma', .99, .998), hp.uniform('init_epsilon', 0.0, 0.5),
                 hp.uniform('n_exploration_episodes', 0, 500),
                 hp.choice('batch_norm', [True, False])
                 ]
        # minimize the values
        env = NormalizedObservationWrapper(gym.make('MountainCar-v0'))
        best = fmin(self._experiment(env), space, algo=tpe.suggest, max_evals=50)
        print("best hyperparameters: alpha={}, alpha_decay={}, gamma={:.3f}, init_epsilon:{}, batch_norm:{}".format(best['alpha'],
            best['alpha_decay'], best['gamma'], best['init_epsilon'], best['batch_norm']))

        # all hyper parameters are know after all experiments
        exp_summary = self._create_experiment_summary()
        root_writer = tf.summary.create_file_writer(log_dir)
        with root_writer.as_default():
            tf.summary.import_event(tf.compat.v1.Event(summary=exp_summary).SerializeToString())

#%%

if __name__ == "__main__":
    unittest.main()
