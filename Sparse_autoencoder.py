import os
import tensorflow as tf
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
mnist = input_data.read_data_sets('./tmp/data', one_hot=True)

def kL_divergence(p, q):
    return p * tf.log(p / q) + (1-p) * tf.log((1-p)/(1-q))

x = tf.placeholder(tf.float32, [None, 28*28])
hidden_layer = partial(tf.layers.dense, activation=tf.nn.elu)
n_inputs = 784
he_init = tf.contrib.layers.variance_scaling_initializer()
beta = 0.2 #sparse weight
p = 0.2 #sparse target
n_hidden1 = 300
n_outputs = n_inputs

with tf.name_scope('model'):
    hidden1 = hidden_layer(x, 300)
    output = hidden_layer(hidden1, n_outputs, activation=None)

with tf.name_scope('training'):
    reconstruction_loss = tf.reduce_mean(tf.square(output-x))
    hidden_act = tf.reduce_mean(hidden1, axis=0) #バッチの平均
    sparsity_loss  = tf.reduce_sum(kL_divergence(p, hidden_act))
    loss = reconstruction_loss + beta * sparsity_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        x_batch, _ = mnist.train.next_batch(150)
        sess.run([training_op], feed_dict={x: x_batch})

        if epoch % 1000 == 0:
            X_test = mnist.test.images
            number = np.random.randint(1, 10000, 1)
            origin_image = np.array(X_test[number[0]]).reshape((28,28))

            X_test_input = X_test[number[0]].reshape(1, 784)
            decode_result = sess.run(output, feed_dict={x: X_test_input})
            decode_image = np.array(decode_result[0]).reshape((28,28))

            fig = plt.figure()
            fig.suptitle('Step {} Result'.format(epoch))
            ax1 = fig.add_subplot(121)
            ax1.imshow(origin_image, cmap='Greys')
            ax1.set_title('original image')

            ax2 = fig.add_subplot(122)
            ax2.imshow(decode_image, cmap='Greys')
            ax2.set_title('decoded image')
            plt.show()
