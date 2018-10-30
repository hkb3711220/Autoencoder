import os
import numpy as np
import tensorflow as tf
from tensorflow.nn import relu
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm

os.chdir(os.path.dirname(__file__))
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

class SegNet(object):

    def __init__(self):
        self.n_epoch = 1
        self.batch_size = 1
        self.NUM_CLASSES = 2

    def encoder_layers(self, inputs, scope):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        inputs = tf.pad(inputs, [[0,0], [2,2], [2,2], [0,0]]) #[32*32*1]

        #Define 3 encoder

        with tf.variable_scope(scope):
            conv1 = self.conv_2d(inputs, 3, 1, inputs.get_shape().as_list()[3], 64) # [32*32*64]
            max_pool1, pool_indice1 = self.max_pool(conv1, 2, 2) # [16*16*64]
            conv2 = self.conv_2d(max_pool1, 3, 1, 64, 64) # [16*16*64]
            max_pool2, pool_indice2 = self.max_pool(conv2, 2, 2) # [8*8*64]
            conv3 = self.conv_2d(max_pool2, 3, 1, 64, 64) # [8*8*64]
            max_pool3, pool_indice3 = self.max_pool(conv3, 2, 2) # [4*4*64]

        return max_pool3, pool_indice1, pool_indice2, pool_indice3

    def decoder_layers(self, inputs, pool_indice1,
                       pool_indice2, pool_indice3, scope):

        with tf.variable_scope(scope):
            UpSample1 = self.Upsampling(inputs, pool_indice3, 3, 64, step=2) # [8*8*64]
            conv4 = self.conv_2d(UpSample1, 3, 1, 64, 64) # [8*8*64]
            UpSample2 = self.Upsampling(conv4, pool_indice2, 3, 64, step=2) # [16*16*64]
            conv5 = self.conv_2d(UpSample2, 3, 1, 64, 64) # [16*16*64]
            UpSample3 = self.Upsampling(conv5, pool_indice1, 3, 64, step=2) # [32*32*64]
            conv6 = self.conv_2d(UpSample3, 3, 1, 64, 64) # [32*32*64]

        """ end of Decode """

        return conv6

    def train(self):

        inputs = tf.placeholder(tf.float32, [None, 28*28])
        encoder_output, pool_indice1, pool_indice2, pool_indice3 = self.encoder_layers(inputs, scope='encoder')
        decoder_output = self.decoder_layers(encoder_output, pool_indice1, pool_indice2, pool_indice3, scope='decoder')
        print(decoder_output)
        logits = tf.reshape(decoder_output, (-1, self.NUM_CLASSES))
        pred = tf.nn.softmax(logits)
        print(logits)
        print(pred)



    def conv_2d(self, inputs, kernel_size, step, num_input, num_output):

        #SegNet 使用的卷积为same 卷积，即卷积后不改变图片大小

        shape = [kernel_size, kernel_size, num_input, num_output]
        kernel = self.create_variable(shape)
        bias = self.create_variable([num_output])
        stride = [1,step,step,1]

        #PREACTIVATION
        conv = tf.nn.conv2d(inputs, kernel, stride, padding='SAME')
        conv_output = relu(batch_norm(conv+bias))

        return conv_output

    def max_pool(self, inputs, kernel, stride):

        return tf.nn.max_pool_with_argmax(inputs,
                                          ksize=[1, kernel, kernel, 1], #Filter [2, 2]
                                          strides=[1, stride, stride, 1],
                                          padding='SAME')

    def Upsampling(self, inputs, pool_indice, kernel_size, output_channels, step=2):

        shape = pool_indice.get_shape().as_list()
        h = shape[1]*2
        w = shape[2]*2
        in_channels = shape[3]

        #valueはconv2dと同じく4次元のTensor。
        #filterは[height, width, output_channels, in_channels] の4次元テンソルで指定しろ、とのこと。
        #output_shapeを指定する必要のある部分がconv2dと異なっており、ここには出力の形状を1次元Tensorで指定するとのこと。
        #stridesはconv2dと同じく、valueのdata_formatを踏まえた指定となる。
        shape = [kernel_size, kernel_size, output_channels, in_channels]
        kernel = self.get_deconv_filter(shape)
        output_shape = [self.batch_size, h, w, in_channels]
        stride = [1,step,step,1]
        #在Decoder 过程中，同样使用same卷积，不过卷积的作用是为upsampling 变大的图像丰富信息，使得在Pooling 过程丢失的信息可以通过学习在Decoder 得到
        return tf.nn.conv2d_transpose(inputs,
                                      kernel,
                                      output_shape=output_shape,
                                      strides=stride,
                                      padding="SAME")

    def get_deconv_filter(self, shape):
        """
         reference: https://github.com/MarvinTeichmann/tensorflow-fcn
         """
         width = shape[0]
         heigh = shape[0]
         f = ceil(width/2.0)
         c = (2 * f - 1 - f % 2) / (2.0 * f)
         bilinear = np.zeros([f_shape[0], f_shape[1]])
         for x in range(width):
             for y in range(heigh):
                 value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                 bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


    def create_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape))


if __name__ == '__main__':

    model = SegNet()
    model.train()
