from examples.reinforce import ReinforceNetwork
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
        self._init_epsilon_list = []
        self._batch_norm_list= []
        self._log_dir = "/tmp/logdir/reinforce_cart_pole"

    def _create_experiment_summary(self):
        alpha_list_val = struct_pb2.ListValue()
        alpha_list_val.extend(self._alpha_list)
        gamma_list_val = struct_pb2.ListValue()
        gamma_list_val.extend(self._gamma_list)
        init_epsilon_list_val = struct_pb2.ListValue()
        init_epsilon_list_val.extend(self._init_epsilon_list)
        batch_norm_list_val = struct_pb2.ListValue()
        batch_norm_list_val.extend(self._batch_norm_list)
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
                                   domain_discrete=gamma_list_val),
                api_pb2.HParamInfo(name='init_epsilon',
                                   display_name='Initial exploration',
                                   type=api_pb2.DATA_TYPE_FLOAT64,
                                   domain_discrete=init_epsilon_list_val),
                api_pb2.HParamInfo(name='batch_norm',
                                   display_name='Batch normalization',
                                   type=api_pb2.DATA_TYPE_BOOL,
                                   domain_discrete=batch_norm_list_val)
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
        alpha, gamma, init_epsilon, batch_norm = args
        hparams = {'alpha':alpha, 'gamma':gamma, 'init_epsilon':init_epsilon, 'batch_norm':batch_norm}
        self._alpha_list.append(alpha)
        self._gamma_list.append(gamma)
        self._init_epsilon_list.append(init_epsilon)
        self._batch_norm_list.append(batch_norm)

        writer = tf.summary.create_file_writer(self._log_dir + "/alpha_{}_gamma_{:.3f}_init_eps{}_bn_{}".format(alpha, gamma, init_epsilon, batch_norm))
        with writer.as_default():
            summary_start = hparams_summary.session_start_pb(hparams=hparams)

            env = CartPoleRewardWrapper(gym.make('CartPole-v1'))
            reinforce = ReinforceNetwork(env, alpha=alpha, gamma=gamma, init_epsilon=init_epsilon, min_epsilon=0.0,
                                         batch_normalization=batch_norm, writer=writer)
            num_episodes = 1000
            episode_lenth = 1000
            for episode in range(num_episodes):
                reinforce.training_episode(num_exploration_episodes=int(num_episodes * 2/3), episode_lenght=episode_lenth)

            average_cmulative_reward = reinforce.evaluate_average_sum_reward(5)
            print('Average sum reward after episode:{} for alpha: {}, gamma: {}, init_eps: {}, batch_norm: {}, reward: {}'.
                  format(episode, alpha, gamma, init_epsilon, batch_norm, average_cmulative_reward))
            summary_end = hparams_summary.session_end_pb(api_pb2.STATUS_SUCCESS)
            tf.summary.scalar('sum_reward', average_cmulative_reward, step=1, description="Average sum reward")
            tf.summary.import_event(tf.compat.v1.Event(summary=summary_start).SerializeToString())
            tf.summary.import_event(tf.compat.v1.Event(summary=summary_end).SerializeToString())

        # hyperopt needs negative value to minimize provided function properly based on fmin (no fmax alternative yet..)
        return -average_cmulative_reward

    def test_reinforce_cart_pole(self):
        log_dir = "/tmp/logdir/reinforce_cart_pole"

        space = [hp.uniform('alpha', 1e-4, 1e-2), hp.uniform('gamma', .8, .99), hp.uniform('init_epsilon', 0.0, 1.0),
                 hp.choice('batch_norm', [True, False])]
        # minimize the values
        best = fmin(self._experiment, space, algo=tpe.suggest, max_evals=50)
        print("best hyperparameters: alpha={}, gamma={:.3f}, init_epsilon:{}, batch_norm:{}".format(best['alpha'],
                                                            best['gamma'], best['init_epsilon'], best['batch_norm']))

        # all hyper parameters are know after all experiments
        exp_summary = self._create_experiment_summary()
        root_writer = tf.summary.create_file_writer(log_dir)
        with root_writer.as_default():
            tf.summary.import_event(tf.compat.v1.Event(summary=exp_summary).SerializeToString())

#%%

if __name__ == "__main__":
    unittest.main()
