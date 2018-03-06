# -*- coding: utf-8 -*-
"""
Generative Adversarial network with the discriminator only
"""

import numpy as np
import tensorflow as tf
from InputTFRecord import inputs
from tryBuildNet import buildNet

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# Discriminator, D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer. Goes from 48x48 -> 24x24
        conv_1, output_dim, output_channels = \
            buildNet.conv_layer('conv_1', x_image, 48, 4, 2, 128, do_pool=True)
        lrelu1 = lrelu(conv_1, 0.2)

        # 2nd hidden layer. Goes from 24x24 -> 12x12
        conv_2, output_dim, output_channels = \
            buildNet.conv_layer('conv_2', lrelu1, 24, 4, 2, 256, do_pool=True)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv_2, training=isTrain), 0.2)

        # 3rd hidden layer. Goes from 12x12 -> 6x6
        conv_3, output_dim, output_channels = \
            buildNet.conv_layer('conv_3', lrelu2, 12, 4, 2, 512, do_pool=True)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv_3, training=isTrain), 0.2)

        # 4th hidden layer. Goes from 6x6 -> 3x3
        conv_4, output_dim, output_channels = \
            buildNet.conv_layer('conv_4', lrelu3, 6, 4, 2, 1024, do_pool=True)
        lrelu4 = lrelu(tf.layers.batch_normalization(conv_4, training=isTrain), 0.2)

        # output layer. Goes from 3x3 -> 1x1
        conv_5, output_dim, output_channels = \
            buildNet.conv_layer('conv_5', lrelu4, 1, 6, 1, 1, do_pool=True)

        return tf.nn.sigmoid(conv_5), conv_5
    
    
inputDim = 48
num_output = 10

#Define the placeholders for the images and labels
# 'None' used to be batch_size << haven't tested None yet
x = tf.placeholder(tf.float32, [None,inputDim**2], name="images")
y_ = tf.placeholder(tf.float32, [None,num_output], name="labels")

"""4 convolution network"""
# placeholder for dropout means we can turn it on during training, turn off during testing
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

with tf.name_scope('reshape'):
    # reshape x to a 4D tensor with second and third dimensions being width/height
    x_image = tf.reshape(x, [-1,inputDim,inputDim,1])

tf.summary.image('input', x_image, 4) # Show 4 examples of output images on tensorboard
