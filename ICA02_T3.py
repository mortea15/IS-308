import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

random.seed(111)
rng = pd.date_range(start='1991', periods=360, freq='M')    # Generate a date range from 1991, 360 periods, monthly
ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
ts.plot(c='b', title='Example Time Series')
plt.show()
print(ts.head(10))

# Get data from a csv file
dataset = np.genfromtxt("data-opptak-1991-2017.csv", delimiter=",")
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

tf.reset_default_graph()

num_periods = 20
inputs = 1
hidden = 100
output = 1

# create variable objects
X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

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

plt.title("Forutsett VS Faktisk", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Faktisk")
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forutsett")
plt.legend(loc="upper left")
plt.xlabel("Tidsperioder")

plt.show()
