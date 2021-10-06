# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 19:30:39 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from gym import wrappers
from datetime import datetime
from q_learning import plot_running_avg, FeatureTransformer

class HiddenLayer:
    def __init__(self, M1, M2, f = T.nnet.relu, use_bias = True, zeros = False):
        if zeros:
            W = np.zeros((M1, M2))
        else:
            W = np.random.randn(M1, M2) * np.sqrt(2 / M1)
        
        self.W = theano.shared(W)
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
    
class PolicyModel:
    def __init__(self, ft, D, hidden_layer_sizes_mean = [], hidden_layer_sizes_var = []):
        self.ft = ft
        self.D = D
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_var = hidden_layer_sizes_var
         
        self.mean_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1, M2)
            self.mean_layers.append(layer)
            M1 = M2
        
        layer = HiddenLayer(M1, 1, lambda x: x, use_bias = False, zeros = True)
        self.mean_layers.append(layer)
        
        self.var_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_var:
            layer = HiddenLayer(M1, M2)
            self.var_layers.append(layer)
            M1 = M2
        
        layer = HiddenLayer(M1, 1, T.nnet.softplus, use_bias = False, zeros = False)
        self.var_layers.append(layer)
        
        params = []
        for layer in (self.mean_layers + self.var_layers):
            params += layer.params
        self.params = params
        
        X = T.matrix('X')
        actions = T.vector('actions')
        advantages = T.vector('advantages')
        
        def get_output(layers):
            Z = X
            for layer in layers:
                Z = layer.forward(Z)
            return Z.flatten()
        
        mean = get_output(self.mean_layers)
        var = get_output(self.var_layers) + 1e-4
        
        self.predict_op = theano.function(
            inputs = [X],
            outputs = [mean, var],
            allow_input_downcast=True
            )
        
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.predict_op(X)
    
    def sample_action(self, X):
        pred = self.predict(X)
        mu = pred[0][0]
        v = pred[1][0]
        a = np.random.randn() * np.sqrt(v) + mu
        return min(max(a, -1), 1)
    
    def copy(self):
        clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
        clone.copy_from(self)
        return clone
    
    def copy_from(self, other):
        for p, q in zip(self.params, other.params):
            v = q.get_value()
            p.set_value(v)
            
    def perturb_params(self):
        for p in self.params:
            v = p.get_value()
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            if np.random.random() < 0.1:
                p.set_value(noise)
            else:
                p.set_value(v + noise)
                
def play_one(env, pmodel, gamma):
    obs = env.reset()
    done = False
    totalreward = 0
    iters = 0
    
    while not done and iters < 2000:
        action = pmodel.sample_action(obs)
        obs, reward, done, info = env.step([action])
        
        totalreward += reward
        iters += 1
        
    return totalreward
        
def play_multiple_episodes(env, T, pmodel, gamma, print_iters = False):
    totalrewards = np.empty(T)
    
    for i in range(T):
        totalrewards[i] = play_one(env, pmodel, gamma)
        
        if print_iters:
            print(i, "avg so far:", totalrewards[:(i+1)].mean())
            
    avg_reward = totalrewards.mean()
    print("avg totalrewards:", avg_reward)
    
    return avg_reward

def random_search(env, pmodel, gamma):
    totalrewards = []
    best_avg_reward = float('-inf')
    best_pmodel = pmodel
    num_ep_per_param_test = 3
    
    for t in range(100):
        tmp_pmodel = best_pmodel.copy()
        tmp_pmodel.perturb_params()
        
        avg_reward = play_multiple_episodes(env, num_ep_per_param_test, tmp_pmodel, gamma)
        
        totalrewards.append(avg_reward)
        
        if avg_reward > best_avg_reward:
            best_pmodel = tmp_pmodel
            best_avg_reward = avg_reward
    return totalrewards, best_pmodel

def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(ft, D, [], [])
    gamma = 0.99
    
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename
        env = wrappers.Monitor(env, monitor_dir)
        
    totalrewards, pmodel = random_search(env, pmodel, gamma)
    
    print("max reward:", np.max(totalrewards))
    
    avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma)  
    
    print("avg reward with best model:", avg_totalrewards)
    
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
        