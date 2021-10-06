# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 21:05:11 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import gym
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
import sys
import os

class FeatureTransformer:
    def __init__(self, env, n_components = 500):
        obs_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(obs_examples)
        
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma = 5.0, n_components = n_components)),
            ('rbf2', RBFSampler(gamma = 2.0, n_components = n_components)),
            ('rbf3', RBFSampler(gamma = 1.0, n_components = n_components)),
            ('rbf4', RBFSampler(gamma = 0.5, n_components = n_components))
            ])
        
        example_features = featurizer.fit_transform(scaler.transform(obs_examples))
                
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)
    
class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate = learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)
            
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        assert(len(result.shape) == 2)
        return result
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])
        
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
def play_one(model, env, eps, gamma):
    done = False
    totalreward = 0
    iters = 0
    observation = env.reset()
    
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_obs = observation
        observation, reward, done, info = env.step(action)
        
        next = model.predict(observation)
        
        G = reward + gamma * np.max(next[0])
        
        model.update(prev_obs, action, G)
        
        totalreward += reward 
        iters += 1
        
    return totalreward

def plot_cost_to_go(env, estimator, num_tiles = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num = num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num = num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, Y, Z,
            rstride = 1, cstride = 1, cmap = matplotlib.cm.coolwarm, vmin = -1.0, vmax = 1.0)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-to-go == -V(s)')
    ax.set_title("Cost-to-go function")
    fig.colorbar(surf)
    plt.show()
    
def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100): (t + 1)].mean()
    
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
    
def main(show_plots = True):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99
    
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename
        env = wrappers.Monitor(env, monitor_dir)
        
    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1 * (0.97 ** n)
        if n == 199:
            print("eps: ", eps)
        
        totalreward = play_one(model, env, eps, gamma)
        totalrewards[n] = totalreward
        if (n + 1) % 100 == 0:
            print("episode: ", n, "total reward: ", totalreward)
            
    print("avg reward for last 100 episodes: ", totalrewards[-100:].mean())
    print("total steps: ", -totalrewards.sum())
    
    if show_plots:
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()
        
        plot_running_avg(totalrewards)
        
        plot_cost_to_go(env, model)
        
    
if __name__ == '__main__':
    main()