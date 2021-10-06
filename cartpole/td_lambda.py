# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:49:54 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
from datetime import datetime
from q_learning import FeatureTransformer
from q_learning_bins import plot_running_avg

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        
    def partial_fit(self, x, y, e, lr = 1e-1):
        self.w += lr * (y - x.dot(self.w)) * e
        
    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)
    
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.ft = feature_transformer
        self.models = []
        
        samples = feature_transformer.transform([env.reset()])
        D = samples.shape[1]
        
        for i in range(env.action_space.n):
            model = SGDRegressor(D)
            self.models.append(model)
            
        self.eligibilities = np.zeros((env.action_space.n, D))
        
    def reset(self):
        self.eligibilities = np.zeros_like(self.eligibilities)
        
    def predict(self, s):
        X = self.ft.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        return result
    
    def update(self, s, a, G, gamma, lambda_):
        X = self.ft.transform([s])
        self.eligibilities *= gamma * lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X, G, self.eligibilities[a])
        
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
def play_one(model, env, eps, gamma, lambda_):
    obs = env.reset()
    done = False
    totalreward = 0
    states_actions_rewards = []
    iters = 0
    model.reset()
    
    while not done and iters < 100000:
        action = model.sample_action(obs, eps)
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        
        if done:
            reward = -300
            
        next = model.predict(obs)
        assert(next.shape == (1, env.action_space.n))
        G = reward + gamma * np.max(next[0])
        model.update(prev_obs, action, G, gamma, lambda_)
        
        states_actions_rewards.append((prev_obs, action, reward))
        
        if reward == 1:
            totalreward += reward
            
        iters += 1
        
    return states_actions_rewards, totalreward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.999
    lambda_ = 0.7
    
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename
        env = wrappers.Monitor(env, monitor_dir)
        
    N = 500
    totalrewards = np.empty(N)
    
    for n in range(N):
        eps = 1.0 / np.sqrt(n+1)
        states_actions_rewards, totalreward = play_one(model, env, eps, gamma, lambda_)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward:", totalrewards[max(0, n-100): (n+1)].mean())
            
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("rewards")
    plt.show()
    
    plot_running_avg(totalrewards)