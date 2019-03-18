import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import random
import math

import argparse

def main(Session):
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dagger_iter', type=int, default=20)

    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')

    args = parser.parse_args()
    env = gym.make(args.envname)

    with open(args.expert_data_file, 'rb') as f:
        expert_data = pickle.loads(f.read())
        print(expert_data.keys())

        expert_obs = expert_data['observations']
        expert_act = expert_data['actions']
        expert_act = np.squeeze(expert_act)

        print("get expert demonstrations", len(expert_obs), " obs / ", len(expert_act), " act")
        print("obs_shape: ", expert_obs.shape)
        print("act_shape: ", expert_act.shape)

    BC = BehaviorCloning(Session ,env)

    Session.run(tf.global_variables_initializer())
    max_steps = args.max_timesteps or env.spec.timestep_limit

    for j in range(args.dagger_iter):

        for i in range(args.epochs):
            for k in range(int(math.ceil(len(expert_obs)/args.batch_size))):
                loss = BC.train(expert_obs[args.batch_size*k:args.batch_size*(k+1)], expert_act[args.batch_size*k:args.batch_size*(k+1)])
                print("data.len: ", len(expert_obs), "dagger iter: ", j, " epochs: ", i , " mn_iter: ", k, " loss: ", loss)

        returns = []
        observations = []
        actions = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = BC.step(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                # env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        print('obs.shape', np.array(observations).shape)
        print('act.shape', np.array(actions).shape)

        observations = np.squeeze(np.array(observations))
        actions = np.squeeze(np.array(actions))

        expert_obs = np.concatenate((expert_obs, observations))
        expert_act = np.concatenate((expert_act, actions))



class BehaviorCloning(object):
    def __init__(self, Session, env):
        self.state_size= env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.hidden_size = 128
        self.build_ph()
        self.model = self.build_model('bc')
        self.loss, self.optimizer = self.build_optimizer()

        self.sess = Session
        self.saver = tf.train.Saver()

    def build_ph(self):
        self.expert_obs = tf.placeholder(tf.float32, shape=[None, self.state_size])
        self.expert_act = tf.placeholder(tf.float32, shape=[None, self.action_size])

    def build_model(self, name):
        with tf.variable_scope(name) as scope:

            w1 = tf.get_variable("w1", shape=[self.state_size, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=[self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            l1 = tf.nn.relu(tf.matmul(self.expert_obs, w1) + b1)

            w2 = tf.get_variable("w2", shape=[self.hidden_size, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape=[self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            w3 = tf.get_variable("w2", shape=[self.hidden_size, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b2", shape=[self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            w4 = tf.get_variable("w_steer", shape=[self.hidden_size, self.action_size], initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            b4 = tf.get_variable("b_steer", shape=[self.action_size], initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            policy = tf.nn.tanh(tf.matmul(l3, w4) + b4)

            scope.reuse_variables()

        return policy

    def build_optimizer(self):
        loss = 0.5 * tf.reduce_mean(tf.square(self.model - self.expert_act))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        return loss, optimizer

    def train(self, expert_obs, expert_act):
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.expert_obs: expert_obs, self.expert_act: expert_act})

        return loss

    def save_model(self):
        checkpoint_dir = 'trained_model'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_model'))

    def load_model(self):
        checkpoint_dir = 'trained_model'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_model'))

    def step(self, obs):
        return self.sess.run(self.model, feed_dict={self.expert_obs: obs})

    def test(self):
        pass


if __name__ == "__main__":
    # config = tf.ConfigProto()
    # If you use a GPU, uncomment
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # config.log_device_placement = False
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6

    with tf.Session() as sess:
        # trainOptions = Options()
        main(Session=sess) # , config=trainOptions)