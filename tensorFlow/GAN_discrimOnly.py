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
            buildNet.conv_layer('conv_1', x_image, 48, 5, 1, 32, do_pool=True)
        lrelu1 = lrelu(conv_1, 0.2)

        # 2nd hidden layer. Goes from 24x24 -> 12x12
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer. Goes from 12x12 -> 6x6
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer. Goes from 6x6 -> 3x3
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer. Goes from 3x3 -> 1x1
        conv5 = tf.layers.conv2d(lrelu4, 1, [3, 3], strides=(1, 1), padding='valid')

        return tf.nn.sigmoid(conv5), conv5
    
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
