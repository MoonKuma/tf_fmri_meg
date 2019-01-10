#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : method.py
# @Author: MoonKuma
# @Date  : 2019/1/9
# @Desc  : implement relative methods here

import numpy as np

# test data
# n_x = 4
# m_x = 10
# n_W = 6
# np.random.seed(0)
# X = np.random.randn(n_x, m_x)
# Y = np.logical_xor(X[0, :] > 0, X[1, 0] > 0).reshape(1, X.shape[1])
# W = np.random.randn(n_W, X.shape[0])
# B = np.random.randn(n_W, 1)
# model_list = ['sigmoid', 'relu']
# compute_table = dict()
# derivative_table = dict()


compute_table = dict()
derivative_table = dict()


# Z = W*X + B
def np_compute_z(X, W, B):
    return np.dot(W, X) + B


# A = activation(Z)
def np_compute_a(Z, model_name):
    return get_compute(model_name)(Z)


# Activation
def get_compute(model_name):
    return compute_table[model_name]


def np_compute_sigmoid(X):
    s = 1/(1+np.exp(-1*X))
    return s


def np_compute_relu(X):
    return np.where(X > 0, X, 0)


# Cost
def np_loss_cross_entropy(Y, A):
    return np.sum(-1*(Y*np.log(A) + (1-Y)*np.log(1-A)),axis=1,keepdims=True)/Y.shape[1]


# Compute Accuracy
def np_accuracy(Y, A):
    A_0 = np.where(A > 0.5, 1, 0)
    return np.sum(np.where(Y == A_0, 1, 0)) / Y.shape[1]


# Derivative
def get_derivative(model_name):
    return derivative_table[model_name]


def np_derivative_sigmoid(X):
    sig_x = np_compute_sigmoid(X)
    return sig_x*(1-sig_x)


def np_derivative_relu(X):
    return np.where(X > 0, 1, 0)


def np_derivative_z(w_uper, dz_uper, z_current, current_model):
    return np.dot(w_uper.T, dz_uper)*get_derivative(current_model)(z_current)


def np_derivative_w(dz_current, A_below, m_x):
    return np.dot(dz_current, A_below.T)/m_x


def np_derivative_b(dz_current, m_x):
    return np.sum(dz_current, axis=1, keepdims=True)/m_x


compute_table['sigmoid'] = np_compute_sigmoid
compute_table['relu'] = np_compute_relu
derivative_table['sigmoid'] = np_derivative_sigmoid
derivative_table['relu'] = np_derivative_relu


