# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 23:29:25 2021

@author: hp
"""
from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import random
import copy
import theano
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.nnet import conv2d
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize

MAX_EXP = 500000
MIN_EXP = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

def downsample_image(A):
    B = A[34:194]
    B = rgb2gray(B)
    
    B = imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')
    return B

def update_state(state, obs):
    obs_small = downsample_image(obs)
    return np.append(state[1:],  np.expand_dims(obs_small, 0), axis = 0)

class ReplayMemory:
    def __init__(self, size=MAX_EXP, frame_height=IM_SIZE, frame_width=IM_SIZE,
                 agent_history_length=4, batch_size =32):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        self.actions = np.empty(self.size, dtype = np.int32)
        self.rewards = np.empty(self.size, dtype = np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype = np.uint8)
        self.terminal_flags = np.empty(self.size, dtype = np.bool)
        
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype = np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype = np.uint8)
        self.indices = np.empty(self.batch_size, dtype = np.int32)
        
    def add_experience(self, action, frame, reward, terminal):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        
    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1: index + 1, ...]
    
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length: index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError("Not enough memories to get a minibatch")
        
        self._get_valid_indices()
        
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
            
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices]
    
def init_filter(shape):
    w = np.random.randn(*shape) *np.sqrt(2.0 / np.prod(shape[1:]))
    return w.astype(np.float32)

def adam(cost, params, lr0=1e-5, beta1=0.9, beta2=0.999, eps=1e-8):
    lr0 = np.float32(lr0)
    beta1 = np.float32(beta1)
    beta2 = np.float32(beta2)
    eps = np.float32(eps)
    one = np.float32(1)
    zero = np.float32(0)
    
    grads = T.grad(cost, params)
    updates = []
    time = theano.shared(zero)
    new_time = time + one
    updates.append((time, new_time))
    lr = lr0*T.sqrt(one - beta2**new_time) / (one - beta1**new_time)
    for p, g in zip(params. grads):
        m = theano.shared(p.get_value() * zero)
        v = theano.shared(p.get_value() * zero)
        new_m = beta1*m + (one - beta1)*g
        new_v = beta2*v + (one - beta2)*g*g
        new_p = p - lr*new_m / (T.sqrt(new_v) + eps)
        updates.append((v, new_v))
        updates.append((m, new_m))
        updates.append((p, new_p))
    return updates

class ConvLayer(object):
    def __init__(self, mi, mo, filtsz=5, stride=2,f=T.nnet.relu):
        sz = (mo, mi, filtsz, filtsz)
        w0 = init_filter(sz)
        self.W = theano.shared(w0)
        b0 = np.zeros(mo, dtype = np.float32)
        self.b = theano.shared(b0)
        self.stride = (stride, stride)
        self.params = [self.W, self.b]
        self.f = f
        
    def forward(self, X):
        conv_out = conv2d(input=X, filters=self.W, subsample=self.stride, border_mode='valid')
        return self.f(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    
class HiddenLayer:
    def __init__(self, M1, M2, f=T.nnet.relu):
        W = np.random.randn(M1, M2) * np.sqrt(2/M1)
        self.W = theano.shared(W.astype(np.float32))
        self.b = theano.shared(np.zeros(M2).astype(np.float32))
        self.params = [self.W, self.b]
        self.f = f
    
    def forward(self, X):
        a = X.dot(self.W) + self.b
        return self.f(a)
    
class DQN:
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes):
        self.K = K
        X = T.ftensor4('X')
        G = T.fvector('G')
        actions = T.ivector('actions')
         
        self.conv_layers = []
        num_input_filters = 4
        current_size = IM_SIZE
        for num_output_filters, filtersz, stride in conv_layer_sizes:
            layer = ConvLayer(num_input_filters, num_output_filters, filtersz, stride)
            current_size = (current_size + stride - 1) // stride
            
        