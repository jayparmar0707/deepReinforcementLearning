# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:14:36 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range
import sys
import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import pandas as pd

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bins(value, bins):
    return np.digitize(x = [value], bins = bins)[0]

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)
        
    def transform(self, obs):
        cart_pos, cart_vel, pole_ang, pole_vel = obs
        return build_state([
            to_bins(cart_pos, self.cart_position_bins),
            to_bins(cart_vel, self.cart_velocity_bins),
            to_bins(pole_ang, self.pole_angle_bins),
            to_bins(pole_vel, self.pole_velocity_bins)
            ])
    
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        state_space = 10 ** env.observation_space.shape[0]
        action_space = env.action_space.n
        self.Q = np.random.uniform(low = -1, high = 1, size = (state_space, action_space))
        
    def predict(self, obs):
        x = self.feature_transformer.transform(obs)
        return self.Q[x]
    
    def update(self, s, a, G):
        x = self.feature_transformer.transform(s)
        self.Q[x, a] += 1e-2 * (G - self.Q[x, a])
        
    def sample_action(self, s, eps):
        p = np.random.random()
        if p < eps:
            return self.env.action_space.sample()
        else:
            y = self.predict(s)
            return np.argmax(y)
        
def play_one_eps(model, eps, gamma):
    obs = env.reset()
    totalrewards = 0
    done = False
    iters = 0
    
    while not done and iters < 10000:
        action = model.sample_action(obs, eps)
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        
        totalrewards += reward
        
        if done and iters < 199:
            reward = -300
        
        G = reward + gamma * (np.max(model.predict(obs)))
        
        model.update(prev_obs, action, G)
        iters += 1
        
    return totalrewards

def plot_running_avg(totalreward):
    N = len(totalreward)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalreward[max(0, i - 100): (i + 1)].mean()
    
    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()
    
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9
    
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = filename + '_videos'
        env = wrappers.Monitor(env, monitor_dir)
        
    
    N = 10000
    totalrewards = np.empty(N)
    
    for n in range(N):
        eps = 1.0/np.sqrt(n + 1)
        totalreward = play_one_eps(model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print(f"episode: {n}, reward: {totalreward}, eps = {eps}")
        
    print("Reward for last 100 eps: ", totalrewards[-100:].mean())
    print("Total steps: ", totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()
    
    plot_running_avg(totalrewards)
        
        