# https://mapr.com/blog/deep-learning-tensorflow/

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from numpy import genfromtxt

random.seed(111)
rng = pd.date_range(start='2000', periods=209, freq='M')
ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
ts.plot(c='b', title='Samordna Opptak Tidsserie')
plt.show()
print(ts.head(10))

# Get data from a csv file
dataset = genfromtxt("data-opptak-1991-2017.csv", delimiter=",")
col1, col2 = dataset.T
col1 = np.delete(col1, 0)  # column header
col2 = np.delete(col2, 0)  # column header

n_samples = col1.shape[0]

ts = pd.Series(col2, index=col1)
ts.plot(c='g', title='Samordna Opptak Tidsserie')
plt.show()
print(ts)

TS = np.array(ts)
num_periods = 4  # in mapr example this is 20
f_horizon = 1  # forecast horizon, one period into the future

x_data = TS[:(len(TS) - (len(TS) % num_periods))]
x_batches = x_data.reshape(-1, 4, 1)  # the same as nr of periods?

y_data = TS[1:(len(TS) - (len(TS) % num_periods)) + f_horizon]
y_batches = y_data.reshape(-1, 4, 1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])
print(y_batches[0:1])
print(y_batches.shape)


def test_data(series, forecast, num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 4, 1)
    testY = TS[-(num_periods):].reshape(-1, 4, 1)
    return testX, testY


X_test, Y_test = test_data(TS, f_horizon, num_periods)
print(X_test.shape)
print(X_test)

# No previous graph objects are running, but this would reset the graphs
tf.reset_default_graph()
# number of periods per vector we are using to predict
# one period ahead
num_periods = 4
# number of vectors submitted
inputs = 1
# number of neurons we will recursively work through
# can be changed to improve accuracy
hidden = 100
# number of output vectors
output = 1

# create variable objects
X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

# create the RNN object
# relu - rectified linear unit (alternatively Sigmoid, Hyberbolic Tangent (Tanh))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)

# choose dynamic over static
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# choose small learning rate to not overshoot the minimum
learning_rate = 0.001

# change the form into a tensor
stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
# specify the type of layer (dense)
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
# shape of results
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

# define the cost function which evaluates the quality of the model
loss = tf.reduce_sum(tf.square(outputs - y))
# gradient descent method
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train the result of the application of the cost function
training_op = optimizer.minimize(loss)

# initialize all the variables
init = tf.global_variables_initializer()

# implementing the model on the training data
# number of iterations or training cycles,
# includes both the FeedForward and Backpropagation
epochs = 1000

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE", mse)

    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)

# Make a plot showing the quality of the prediction
plt.title("Forutsett vs Faktisk", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Faktisk")
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forutsett")
plt.legend(loc="upper left")
plt.xlabel("Tidsperioder")
plt.show()
