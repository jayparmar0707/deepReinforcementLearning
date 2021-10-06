# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:30:49 2021

@author: hp
"""

from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow.compat.v1 as tf
import q_learning
tf.disable_v2_behavior()

class SGDRegressor:
    def __init__(self, D):
        print("Hello Tensorflow")
        lr = 0.1
        
        self.w = tf.Variable(tf.random.normal(shape = (D, 1)), name = 'w')
        self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'X')
        self.Y = tf.placeholder(tf.float32, shape = (None, ), name = 'Y')
        
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)
        
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat
        
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)
        
    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict = {self.X: X, self.Y: Y})
        
    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict = {self.X: X})
    
if __name__ == '__main__':
    q_learning.SGDRegressor = SGDRegressor
    q_learning.main()