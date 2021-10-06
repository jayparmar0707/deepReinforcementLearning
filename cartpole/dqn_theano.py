# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 22:12:24 2021

@author: hp
"""

from __future__ import print_function, division
from builtins import range

import os
import sys
import gym
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning import plot_running_avg

global_iters = 0

def adam(cost, params, lr0=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    grads = T.grad(cost, params)
    updates = []
    time = theano.shared(0)
    new_time = time + 1
    updates.append((time, new_time))
    lr = lr0*T.sqrt(1 - beta2**new_time) / (1 - beta1**new_time)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        new_m = beta1*m + (1 + beta1)*g
        new_v = beta2*v + (1 + beta2)*g*g
        new_p = p - lr*new_m/ (T.sqrt(new_v) + eps)
        updates.append((m, new_m))
        updates.append((v, new_v))
        updates.append((p, new_p))
    return updates

class HiddenLayer:
    def __init__(self, M1, M2, f=T.tanh, use_bias = True):
        self.W = theano.shared(np.random.randn(M1, M2) * np.sqrt(2 / M1))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = theano.shared(np.zeros(M2))
            self.params += [self.b]
        self.f = f
    
    def forward(self, X):
        if self.use_bias:
            a = X.dot(self.W) + self.b
        else:
            a = X.dot(self.W)
        return self.f(a)
    
class DQN:
    def __init__(self, D, K, hidden_layer_sizes, gamma, max_exp = 10000, min_exp = 100, batch_sz = 32):
        self.K = K
        lr = 1e-2
        mu = 0.
        decay = 0.99
        
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
            
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params
            
        X = T.matrix('X')
        G = T.vector('G')
        actions = T.ivector('actions')
        
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
        
        Y_hat = Z
        
        selected_action_values = Y_hat[T.arange(actions.shape[0]), actions]
        cost = T.sum((G - selected_action_values)**2)
        
        updates = adam(cost, self.params)
        
        self.train_op = theano.function(
            inputs = [X, G, actions],
            updates = updates,
            allow_input_downcast=True
            )
        
        self.predict_op = theano.function(
            inputs = [X],
            outputs = Y_hat,
            allow_input_downcast = True
            )
        
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_exp = max_exp
        self.min_exp = min_exp
        self.batch_sz = batch_sz
        self.gamma = gamma
        
    def copy_from(self, other):
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            a = q.get_value()
            p.set_value(a)
            
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.predict_op(X)
    
    def train(self, target_network):
        if len(self.experience['s']) < self.min_exp:
            return
        
        idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace = False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis = 1)
        targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]
        
        self.train_op(states, targets, actions)
        
    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s'])>= self.max_exp:
            self.experience['s'].pop(0)
            self.experience['r'].pop(0)
            self.experience['a'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
            
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)
        
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])
        
def play_one(env, model, tmodel, eps, gamma, copy_period):
    global global_iters
    obs = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        a = model.sample_action(obs, eps)
        prev_obs = obs
        obs, reward, done, info = env.step(a)
        
        totalreward += reward
        if done:
            reward = -200
        
        model.add_experience(prev_obs, a, reward, obs, done)
        model.train(tmodel)
        
        iters += 1
        global_iters += 1
        
        if global_iters % copy_period == 0:
            tmodel.copy_from(model)
            
    return totalreward

def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50
    
    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [200, 200]
    model = DQN(D, K, sizes, gamma)
    tmodel = DQN(D, K, sizes, gamma)
    
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename
        env = wrappers.Monitor(env, monitor_dir)
        
    
    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
        totalrewards[n] = totalreward
        
        if n % 100 == 0:
            print("episode:", n, "eps:", eps, "total reward:", totalreward, "avg reward:", totalrewards[max(0, n-100):(n+1)].mean())
            
    print("avg reward for last 100 episodes:", totalrewards[-100].mean())
    print("total steps:", totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("rewards")
    plt.show()
    
    plot_running_avg(totalrewards)
    
if __name__ == '__main__':
    main()
            