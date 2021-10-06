# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:59:26 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_ep(env, params):
    obs = env.reset()
    t = 0
    done = False
    while not done and t < 10000:
        t += 1
        action = get_action(obs, params)
        obs, reward, done, info = env.step(action)
        
        if done:
            break
        
    return t

def play_multiple_eps(env, t, params):
    episode_lengths = np.empty(t)
    
    for i in range(t):
        episode_lengths[i] = play_one_ep(env, params)
        
    avg_length = episode_lengths.mean()
    print("avg length: ", avg_length)
    return avg_length

def random_search(env):
    best = 0
    params = None
    episode_lengths = []
    
    for i in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_eps(env, 100, new_params)
        episode_lengths.append(avg_length)
        
        if avg_length > best:
            best = avg_length
            params = new_params
            
    return episode_lengths, params 
        
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()
    
    env = wrappers.Monitor(env, 'my_awesome_dir')
    print("Final run: ", play_one_ep(env, params))
    