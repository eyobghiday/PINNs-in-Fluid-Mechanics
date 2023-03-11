# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# on progress
"""
Created on Thu Feb 23 14:05:18 2023

@author: eyob
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

xu = np.array([1.000, 0.983, 0.956, 0.917, 0.867, 0.806, 0.733,
              0.650, 0.556, 0.456, 0.350, 0.242, 0.133, 0.027, 0.000])
yu = np.array([0.000, 0.053, 0.103, 0.150, 0.192, 0.229, 0.259,
              0.283, 0.299, 0.308, 0.309, 0.302, 0.288, 0.266, 0.260])
xl = np.array([1.000, 0.983, 0.956, 0.917, 0.867, 0.806, 0.733,
              0.650, 0.556, 0.456, 0.350, 0.242, 0.133, 0.027, 0.000])
yl = np.array([0.000, -0.053, -0.103, -0.150, -0.192, -0.229, -0.259, -
              0.283, -0.299, -0.308, -0.309, -0.302, -0.288, -0.266, -0.260])
x = np.concatenate([xu, xl[::-1]])
y = np.concatenate([yu, yl[::-1]])

# Define flow conditions
u_inf = 1.0
nu = 0.001
Re = u_inf * 1.0 / nu  # assuming IIT Chicago reynolds

# Generate training data
n_data = 500
n_airfoil = 100
x_airfoil = np.linspace(0, 1, n_airfoil)
y_airfoil = np.zeros(n_airfoil)
X_airfoil = np.concatenate([x_airfoil, x_airfoil[::-1]])
Y_airfoil = np.concatenate([y_airfoil, -y_airfoil[::-1]])

X_bc = np.random.uniform(low=-0.5, high=1.5, size=(n_data, 1))
Y_bc = np.random.uniform(low=-0.5, high=0.5, size=(n_data, 1))
X_int = np.random.uniform(low=-0.5, high=1.5, size=(n_data, 1))
Y_int = np.random.uniform(low=-0.5, high=0.5, size=(n_data, 1))

# I will utilise tensors here
# Converting to tensors

X_airfoil_tf = tf.convert_to_tensor(X_airfoil.reshape(-1, 1), dtype=tf.float32)
Y_airfoil_tf = tf.convert_to_tensor(Y_airfoil.reshape(-1, 1), dtype=tf.float32)
X_bc_tf = tf.convert_to_tensor(X_bc, dtype=tf.float32)
Y_bc_tf = tf.convert_to_tensor(Y_bc, dtype=tf.float32)
X_int_tf = tf.convert_to_tensor(X_int, dtype=tf.float32)
Y_int_tf = tf.convert_to_tensor(Y_int, dtype=tf.float32)

.... ..... .....
