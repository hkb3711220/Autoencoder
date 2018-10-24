import os
import tensorflow as tf
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
mnist = input_data.read_data_sets('./tmp/data', one_hot=True)

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs
scale = 0.0001
n_epoch = 10000

x = tf.placeholder(tf.float32, [None, n_inputs])
hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale))
he_init = tf.contrib.layers.variance_scaling_initializer()

hidden_layer1 = hidden_layer(x, n_hidden1)
hidden_layer2 = hidden_layer(hidden_layer1, n_hidden2)
hidden_layer3 = hidden_layer(hidden_layer2, n_hidden3)
output = hidden_layer(hidden_layer3, n_outputs, activation=None)

mse = tf.reduce_mean(tf.square(output-x))
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([mse] + reg_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        X_batch, _ = mnist.train.next_batch(150)
        sess.run([training_op], feed_dict={x:X_batch})

    X_test = mnist.test.images
    number = np.random.randint(1, 10000, 1)
    origin_image = np.array(X_test[number[0]]).reshape((28,28))

    X_test_input = X_test[number[0]].reshape(1, 784)
    decode_result = sess.run(output, feed_dict={x: X_test_input})
    decode_image = np.array(decode_result[0]).reshape((28,28))

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(origin_image, cmap='Greys')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(decode_image, cmap='Greys')
    plt.axis('off')
    plt.show()
