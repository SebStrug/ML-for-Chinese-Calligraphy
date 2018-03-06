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
            #48 is the input dimension, patch size of 4, 
            buildNet.conv_layer('conv_1', x_image, 48, 4, 2, 128, do_pool=True)
        lrelu1 = tf.nn.leaky_relu(conv_1)

        # 2nd hidden layer. Goes from 24x24 -> 12x12
        conv_2, output_dim, output_channels = \
            buildNet.conv_layer('conv_2', lrelu1, 24, 4, 2, 256, do_pool=True)
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_2, training=isTrain))

        # 3rd hidden layer. Goes from 12x12 -> 6x6
        conv_3, output_dim, output_channels = \
            buildNet.conv_layer('conv_3', lrelu2, 12, 4, 2, 512, do_pool=True)
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_3, training=isTrain))

        # 4th hidden layer. Goes from 6x6 -> 3x3
        conv_4, output_dim, output_channels = \
            buildNet.conv_layer('conv_4', lrelu3, 6, 4, 2, 1024, do_pool=True)
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv_4, training=isTrain))

        # output layer. Goes from 3x3 -> 1x1
        conv_5, output_dim, output_channels = \
            buildNet.conv_layer('conv_5', lrelu4, 1, 6, 1, 1, do_pool=True)

        return tf.nn.sigmoid(conv_5), conv_5
    
    
inputDim = 48
num_output = 10

batch_size = 128
num_epochs = 100

# load data
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data": True}
localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0'
train_tfrecord_filename = localPath+'\\train'+str(num_output)+'.tfrecords'
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,\
                                                  batch_size,num_epochs, **train_kwargs)

#Define the placeholders for the images and labels
x = tf.placeholder(tf.float32, [None,inputDim**2], name="images")
#y_ = tf.placeholder(tf.float32, [None,num_output], name="labels") # holds the labels
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100)) # holds random numbers for the generator
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') # for the dropout layer
isTrain = tf.placeholder(dtype=tf.bool) # boolean for training or not

with tf.name_scope('reshape'):
    # reshape x to a 4D tensor with second and third dimensions being width/height
    # [None, inputDim, inputDim, 1] didn't work
    x_image = tf.reshape(x, [-1,inputDim,inputDim,1])

tf.summary.image('input', x_image, 4) # Show 4 examples of output images on tensorboard

D_real, D_real_logits = discriminator(x, isTrain)



