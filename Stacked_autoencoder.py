import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.chdir(os.path.dirname(__file__))
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

def create_variable(shape):
    return tf.Variable(tf.truncated_normal(shape))

def stacked_autoencoder(n_hidden1, n_hidden2):
　　#phase1 と phase2 分けて訓練しているから。
　　#phase1 operation: 入力層 →　隠れ層1 →　出力層
　　#phase2 operation: 隠れ層1 →　コーティング　→　隠れ層2
    x = tf.placeholder(tf.float32, [None, 28*28])
    n_inputs = 784

    activation = tf.nn.elu
    init = tf.contrib.layers.variance_scaling_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    weights1_init = init([n_inputs, n_hidden1])
    weights2_init = init([n_hidden1, n_hidden2])

    weights1 = tf.Variable(weights1_init, dtype=tf.float32)
    weights2 = tf.Variable(weights2_init, dtype=tf.float32)
    weights3 = create_variable([n_hidden2, n_hidden1])
    weights4 = create_variable([n_hidden1, n_inputs])

    bias1 = create_variable([n_hidden1])
    bias2 = create_variable([n_hidden2])
    bias3 = create_variable([n_hidden1])
    bias4 = create_variable([n_inputs])

    hidden1 = activation(tf.matmul(x, weights1) + bias1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + bias2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + bias3)
    output = tf.matmul(hidden1, weights4) + bias4

    optimizer = tf.train.AdamOptimizer(0.001)
    with tf.variable_scope('phase1'):
        phase1_output = tf.matmul(hidden1, weights4) + bias4
        phase1_loss = tf.reduce_mean(tf.square(phase1_output - x))
        phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
        phase1_total_loss = phase1_loss + phase1_reg_loss
        phase1_training_op = optimizer.minimize(phase1_total_loss)

    with tf.variable_scope('phase2'):
        phase2_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
        phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
        phase2_total_loss = phase2_loss + phase2_reg_loss
        phase2_train_list = [weights2, bias2, weights3, bias3]
        phase2_training_op = optimizer.minimize(phase2_total_loss, var_list=phase2_train_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10000):
            x_batch, _ = mnist.train.next_batch(50)
            sess.run([phase1_training_op, phase2_training_op], feed_dict={x: x_batch})

            if epoch % 100 == 0:
                X_test = mnist.test.images
                number = np.random.randint(1, 10000, 1)
                origin_image = np.array(X_test[number[0]]).reshape((28,28))

                X_test_input = X_test[number[0]].reshape(1, 784)
                decode_result = sess.run(output, feed_dict={x: X_test_input})
                decode_image = np.array(decode_result[0]).reshape((28,28))

                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax1.imshow(origin_image, cmap='Greys')
                ax1.set_title('original image')

                ax2 = fig.add_subplot(122)
                ax2.imshow(decode_image, cmap='Greys')
                ax2.set_title('decoded image')
                plt.show()
                plt.close()

if __name__ == '__main__':
    stacked_autoencoder(300, 150)
