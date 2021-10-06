# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:46:43 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range
import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    
    out = env.reset()
    
    box = env.observation_space
    
    act = env.action_space
    
    steps = []
    
    for it in range(100):
        done = False
        n_steps = 0
        while not done:
            obs, reward, done, info = env.step(np.random.choice(env.action_space.n))
            if (it + 1) % 10 == 0:
                print(it + 1)
                print(f"{n_steps}. obs = {obs}, r = {reward}, done = {done}")
            n_steps += 1
        steps.append(n_steps)
        
    print(steps)
    print(np.mean(steps))