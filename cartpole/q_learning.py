# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:31:12 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from q_learning_bins import plot_running_avg

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1
        
    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)
        
    def predict(self, X):
        return X.dot(self.w)
    
class FeatureTransformer:
    def __init__(self, env):
        obs_samples = np.random.random((20000, 4)) * 2 - 1
        scaler = StandardScaler()
        scaler.fit(obs_samples)
        
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma = 0.05, n_components = 1000)),
            ('rbf2', RBFSampler(gamma = 1.0, n_components = 1000)),
            ('rbf3', RBFSampler(gamma = 0.5, n_components = 1000)),
            ('rbf4', RBFSampler(gamma = 0.1, n_components = 1000))
            ])
        feature_examples = featurizer.fit_transform(scaler.transform(obs_samples))
        
        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)
    
class Model:
    def __init__(self, env, ft):
        self.env = env
        self.feature_transformer = ft
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(ft.dimensions)
            self.models.append(model)
            
    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])
        
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
def play_one(env, model, eps, gamma):
    totalreward = 0
    iters = 0
    done = False
    obs = env.reset()
    
    while not done and iters < 2000:
        action = model.sample_action(obs, eps)
        prev_obs = obs
        
        obs, reward, done, info = env.step(action)
        
        if done:
            reward = -200
            
        next = model.predict(obs)
        assert(next.shape == (1, env.action_space.n))
        G = reward + gamma * np.max(next)
        model.update(prev_obs, action, G)
        
        if reward == 1:
            totalreward += reward
            
        iters += 1
        
    return totalreward

def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename
        env = wrappers.Monitor(env, monitor_dir)
        
    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward(last 100):", totalrewards[max(0, n - 100): (n + 1)].mean())
         
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(totalrewards)
    
if __name__ == '__main__':
    main()