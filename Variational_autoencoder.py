import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

class VAE(object):

    def __init__(self, n_hidden1, n_hidden2, is_training):

        x=tf.placeholder(tf.float32, shape=[None, 28 * 28])
        n_inputs = 28 * 28
        self.n_hidden1 = n_hidden1 #h
        self.n_hidden2 = n_hidden2 #Z
        self.n_hidden3 = n_hidden1 #h"
        self.n_outputs = n_inputs
        self.is_traning = is_training

        intializer = tf.contrib.layers.variance_scaling_initializer()
        dense_layer = partial(tf.layers.dense,
                              kernel_initializer=intializer)

        #Encoder
        with tf.variable_scope('encoder'):
            hidden1 = dense_layer(x, self.n_hidden1, activation=tf.nn.tanh) # h= tanh(w0*x + b0)
            hidden2_mean = dense_layer(hidden1, self.n_hidden2, activation=None)
            hidden2_gamma = dense_layer(hidden1, self.n_hidden2, activation=None) #log gamman
            hidden2 = self._hidden2(hidden2_mean, hidden2_gamma)

        with tf.variable_scope('decoder'):
            hidden3 = dense_layer(hidden2, self.n_hidden3, activation=tf.nn.tanh) # h"= tanh(w3*x + b3)
            logits = dense_layer(hidden3, self.n_outputs, activation=None) # W4h' + W4
            self.outputs = tf.nn.sigmoid(logits) # y = sigmoid(logits)

        with tf.variable_scope('train'):
            loss = self._loss(x, logits, hidden2_mean, hidden2_gamma)
            optimizer = tf.train.AdamOptimizer(0.001)
            training_op = optimizer.minimize(loss)


    def _hidden2(self, hidden_mean, hidden_gamma):
        noise = tf.random_normal(tf.shape(hidden_gamma), dtype=tf.float32)
        return hidden_mean + tf.exp(0.5 * hidden_gamma) * noise

    def _loss(self, x, logits, hidden_mean, hidden_gamma):
        recontruction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits))
        latent_loss = -0.5 * tf.reduce_sum(1 + hidden_gamma - hidden_mean - tf.exp(hidden_gamma))
        return recontruction_loss + latent_loss

if __name__ == '__main__':
    VAE = VAE(300, 20, True)
