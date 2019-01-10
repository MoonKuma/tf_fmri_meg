#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: MoonKuma
# @Date  : 2019/1/9
# @Desc  : the main model, this model is built without the help of tf

import numpy as np
from method import *

# load_data
def load_data(file_name):
    data_array = np.loadtxt(file_name, skiprows=1, delimiter=',')
    sift_array = data_array[np.where(data_array[:, 4] < 2)]
    return sift_array


# batch generator
def batch_generator(data_array, n_batch, total_batch):
    if n_batch > data_array.shape[0]:
        msg = 'number of  batch should be smaller than the length of data'
        print(msg)
        raise RuntimeError
    batch_id = 0
    while batch_id < total_batch:
        np.random.shuffle(data_array)
        arr = data_array[0:n_batch, :]
        x, y = arr[:, 0:data_array.shape[1]-1], arr[:,data_array.shape[1]-1]
        batch_id += 1
        yield x.T, y.reshape(1, y.shape[0])


# random initialize
def random_initial(np_array):
    shape = np_array.shape
    return np.random.randn(shape[0], shape[1])*0.01


# compile model 4(input)*10(hidden1, layer1, relu)*5(hidden2, layer2, relu)*1(output, layer3, sigmoid)
def test_model(data_array, n_batch, total_batch):
    x_shape = (data_array.shape[1]-1, n_batch)
    data_generator = batch_generator(data_array, n_batch, total_batch)
    learning_rate = 0.001
    w1 = random_initial(np.zeros((10, x_shape[0])))
    b1 = np.zeros((10, 1))
    w2 = random_initial(np.zeros((5, 10)))
    b2 = np.zeros((5, 1))
    w3 = random_initial(np.zeros((1, 5)))
    b3 = np.zeros((1, 1))
    batch_id = 0
    loss = -1
    for x,y in data_generator:
        # batch id
        batch_id +=1
        # forward
        Z1 = np_compute_z(x, w1, b1)
        A1 = np_compute_a(Z1,'relu')
        Z2 = np_compute_z(A1, w2, b2)
        A2 = np_compute_a(Z2,'relu')
        Z3 = np_compute_z(A2, w3, b3)
        A3 = np_compute_a(Z3,'sigmoid')
        # loss
        loss = np_loss_cross_entropy(y, A3)
        accuracy= np_accuracy(y, A3)
        print('Batch [', batch_id, '], loss:', loss, ',accuracy:', accuracy)
        # backward
        dZ3 = A3 - y
        dw3 = np_derivative_w(dZ3, A2, n_batch)
        db3 = np_derivative_b(dZ3, n_batch)
        dZ2 = np_derivative_z(w3, dZ3, Z2, 'relu')
        dw2 = np_derivative_w(dZ2, A1, n_batch)
        db2 = np_derivative_b(dZ2, n_batch)
        dZ1 = np_derivative_z(w2, dZ2, Z1, 'relu')
        dw1 = np_derivative_w(dZ1, x, n_batch)
        db1 = np_derivative_b(dZ1, n_batch)
        # update (learn)
        w3 -= learning_rate * dw3
        b3 -= learning_rate * db3
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
    # save model
    print('Finish at Batch [', batch_id, '], loss:', loss, ',accuracy:', accuracy)
    model = [[w1, b1, 'relu'], [w2, b2, 'relu'], [w3, b3, 'sigmoid']]
    # print('model', model)
    return model



# compile model 4(input)*10(hidden1, layer1, relu)*1(output, layer3, sigmoid)
def test_model2(data_array, n_batch, total_batch):
    x_shape = (data_array.shape[1]-1, n_batch)
    data_generator = batch_generator(data_array, n_batch, total_batch)
    learning_rate = 0.01
    w1 = random_initial(np.zeros((10, x_shape[0])))
    b1 = np.zeros((10, 1))
    w2 = random_initial(np.zeros((1, 10)))
    b2 = np.zeros((1, 1))
    batch_id = 0
    loss = -1
    for x,y in data_generator:
        # batch id
        batch_id +=1
        # forward
        Z1 = np_compute_z(x, w1, b1)
        A1 = np_compute_a(Z1,'relu')
        Z2 = np_compute_z(A1, w2, b2)
        A2 = np_compute_a(Z2,'sigmoid')

        # loss
        loss = np_loss_cross_entropy(y, A2)
        accuracy= np_accuracy(y, A2)
        print('Batch [', batch_id, '], loss:', loss, ',accuracy:', accuracy)
        # backward
        dZ2 = A2 - y
        dw2 = np_derivative_w(dZ2, A1, n_batch)
        db2 = np_derivative_b(dZ2, n_batch)
        dZ1 = np_derivative_z(w2, dZ2, Z1, 'relu')
        dw1 = np_derivative_w(dZ1, x, n_batch)
        db1 = np_derivative_b(dZ1, n_batch)

        # update (learn)
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    # save model
    print('Finish at Batch [', batch_id, '], loss:', loss, ',accuracy:', accuracy)
    model = [[w1, b1, 'relu'], [w2, b2, 'sigmoid']]
    # print('model', model)
    return model


def predict(model, X, Y):
    A = X
    z = None
    for layers in model:
        w = layers[0]
        b = layers[1]
        model_name = layers[2]
        z = np_compute_z(A, w, b)
        A = np_compute_a(z, model_name)
    accuracy = np_accuracy(y, A2)


data_array = load_data('iris_training.csv')
print('data_array.shape', data_array.shape)
test_model2(data_array, 50, 1000)

# model 2 reach 100% accuracy(train) with current setting
# model 1 may be two complicated
# also learning rate need to be set as 0.01












# # test
# # reference/tf_basic_exercise/dnn_without_tf/iris_test.csv
# data_array = load_data('iris_test.csv')
# print('data_array.shape',data_array.shape)
# n_batch = 20
# batch_count = 5
# generator = batch_generator(data_array, n_batch, batch_count)
# w = np.zeros((5,10))
#
# for x, y in generator:
#     print('x', x)
#     print('y', y)
