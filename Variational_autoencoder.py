import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('tmp/data', one_hot=True)



def _hidden2(hidden_mean, hidden_gamma):
    noise = tf.random_normal(tf.shape(hidden_gamma), dtype=tf.float32)
    return hidden_mean + tf.exp(0.5 * hidden_gamma) * noise


def _loss(x, logits, hidden_mean, hidden_gamma):
    recontruction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits))
    latent_loss = -0.5 * tf.reduce_sum(1 + hidden_gamma - hidden_mean - tf.exp(hidden_gamma))
    return recontruction_loss + latent_loss

n_inputs = 28 * 28
n_hidden1 = 150
n_hidden2 = 20
n_hidden3 = n_hidden1
n_outputs = n_inputs
x=tf.placeholder(tf.float32, shape=[None, 28 * 28])
intializer = tf.contrib.layers.variance_scaling_initializer()
dense_layer = partial(tf.layers.dense, kernel_initializer=intializer)

    #Encoder
with tf.variable_scope('encoder'):
    hidden1 = dense_layer(x, n_hidden1, activation=tf.nn.tanh) # h= tanh(w0*x + b0)
    hidden2_mean = dense_layer(hidden1, n_hidden2, activation=None)
    hidden2_gamma = dense_layer(hidden1, n_hidden2, activation=None) #log gamman
    hidden2 = _hidden2(hidden2_mean, hidden2_gamma)

with tf.variable_scope('decoder'):
    hidden3 = dense_layer(hidden2, n_hidden3, activation=tf.nn.tanh) # h"= tanh(w3*x + b3)
    logits = dense_layer(hidden3, n_outputs, activation=None) # W4h' + W4
    outputs = tf.nn.sigmoid(logits) # y = sigmoid(logits)

with tf.variable_scope('train'):
    loss = _loss(x, logits, hidden2_mean, hidden2_gamma)
    optimizer = tf.train.AdamOptimizer(0.001)
    training_op = optimizer.minimize(loss)

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            X_batch, _ = mnist.train.next_batch(150)
            sess.run(training_op, feed_dict={x: X_batch})

        codings_rnd = np.random.normal(size=[60, n_hidden2])
        outputs_val = outputs.eval(feed_dict={hidden2: codings_rnd})

    for iteration in range(60):
        plt.subplot(60, 10, iteration+1)
        plt.imshow(outputs_val[iteration].reshape([28,28]), cmap="Greys", interpolation="nearest")
        plt.axis('off')

    plt.show()

