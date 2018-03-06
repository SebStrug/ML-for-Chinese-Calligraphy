# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:26:24 2018

@author: Sebastian
"""

import tensorflow as tf

#input_dim = 48
#inputDim = 48
#num_output = 10
#
##Define the placeholders for the images and labels
## 'None' used to be batch_size << haven't tested None yet
#x = tf.placeholder(tf.float32, [None,inputDim**2], name="images")
#y_ = tf.placeholder(tf.float32, [None,num_output], name="labels")
#
#with tf.name_scope('reshape'):
#    # reshape x to a 4D tensor with second and third dimensions being width/height
#    x_image = tf.reshape(x, [-1,inputDim,inputDim,1])
#
#tf.summary.image('input', x_image, 4) # Show 4 examples of output images on tensorboard

class buildNet(object):
    #build a graph here
    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        tf.summary.histogram("weights", initial)
        return tf.Variable(initial)

    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        tf.summary.histogram("biases", initial)
        return tf.Variable(initial)
    
    def conv2d(x, W, stride_input, padding_type):
        """conv2d returns a 2d convolution layer with full stride."""  
        #stride_input in form [2,2], or [1,1]
        return tf.nn.conv2d(x, W, strides=[1] + stride_input + [1], padding = padding_type)
    
    def deconv2d(x, W, output_dim, stride_input, padding_type):
        """deconv2d returns a 2d de-convolution layer with full stride."""    
        #stride_input in form [2,2], or [1,1]
        return tf.nn.conv2d_transpose(x, W, output_shape = tf.constant([output_dim,output_dim], tf.float32), \
                    strides=[1]+stride_input+[1], padding = padding_type)
    
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    def conv_layer(conv_name, prev_layer, input_dim, patch_size, stride, input_channels, \
                   output_channels, do_pool=False, padding_type = 'SAME'):
        """ e.g. name = 'conv-1', x = x_image, patch_size = [5,5], stride = [1,1],
                    input_channels = 32, output_channels = 64"""
        with tf.name_scope(conv_name):
            """First convolution layer, maps one greyscale image to 32 feature maps"""
            # patch size of YxY, 1 input channel, Z output channels (features)
            print("\nBuilding a convolution layer...")
            print("Name: {}, Weight shape: [{},{},{},{}], Bias shape: [{}], Stride: {}, Output channels: {}".\
                  format(conv_name, patch_size, patch_size, input_channels,\
                         output_channels, output_channels, [1]+stride+[1], output_channels))
            print("Doing a pool: {}".format(do_pool))
            W_conv_input = [patch_size,patch_size,input_channels,output_channels]
            W_conv = buildNet.weight_variable(W_conv_input)
            # bias has a component for each output channel (feature)
            b_conv = buildNet.bias_variable([output_channels])
            # convolve x with the weight tensor, add bias and apply ReLU function
            h_conv = tf.nn.relu(buildNet.conv2d(prev_layer, W_conv, stride, padding_type) + b_conv)
            tf.summary.histogram("activations", h_conv)
            print("Output dimension: {}".format(h_conv.shape))
        if do_pool == True:
            pool_name = conv_name + '_pool'
            with tf.name_scope(pool_name):
                """Pooling layer, downsamples by 2x"""
                print("Pooling...")
                h_pool = buildNet.max_pool_2x2(h_conv)
                output_dim = h_pool.shape[1] #can be index 1 or 2
                #return the pooling layer if we did a pool
                return h_pool, output_dim, output_channels
        else: #return the convolutional layer if we didn't do a pool
            output_dim = h_conv.shape[1] #can be index 1 or 2   
            return h_conv, output_dim, output_channels
        
    def deconv_layer(deconv_name, prev_layer, input_dim, patch_size, stride, input_channels, \
                     output_channels, padding_type = 'SAME'):
        with tf.name_scope(deconv_name):
            """First convolution layer, maps one greyscale image to 32 feature maps"""
            # patch size of YxY, 1 input channel, Z output channels (features)
            print("\nBuilding a DE-convolution layer...")
            print("Name: {}, Weight shape: [{},{},{},{}], Bias shape: [{}], Stride: {}, Output channels: {}".\
                  format(deconv_name, patch_size, patch_size, input_channels,\
                         output_channels, output_channels, [1]+stride+[1], output_channels))
            W_conv_input = [patch_size,patch_size,input_channels,output_channels]
            W_conv = buildNet.weight_variable(W_conv_input)
            # bias has a component for each output channel (feature)
            b_conv = buildNet.bias_variable([output_channels])
            # convolve x with the weight tensor, add bias and apply ReLU function
            h_conv = tf.nn.relu(buildNet.deconv2d(prev_layer, W_conv, \
                                        input_dim, stride, padding_type) + b_conv)
            tf.summary.histogram("activations", h_conv)
            print("Output dimension: {}".format(h_conv.shape))
        output_dim = h_conv.shape[1] #can't do a pool in a deconv_layer
        return h_conv, output_dim, output_channels 
               
    def fc_layer(fc_name, input_layer, input_dim, input_features, output_channel, do_pool=False):
        with tf.name_scope(fc_name):
            """Fully connected layer 1, after 2 rounds of downsampling, our 28x28 image
            is reduced to 7x7x64 feature maps, map this to 1024 features"""
            # 7*7 image size *64 inputs, fc layer has 1024 neurons
            print("\nBuilding a fully connected layer...")
            print("Name: {}, Weight shape: [{}*{}*{},{}], Bias shape: [{}]".\
                  format(fc_name,input_dim,input_dim,input_features,output_channel, output_channel))
            print("Doing a pool: {}".format(do_pool))
            
            W_fc = buildNet.weight_variable([input_dim * input_dim * input_features, output_channel])
            b_fc = buildNet.bias_variable([output_channel])
            
            if do_pool == True:
                print("Pooling...")
                # reshape the pooling layer from 7*7 (*64 inputs) to a batch of vectors
                h_pool_flat = tf.reshape(input_layer, [-1, input_dim * input_dim * input_features])
                # do a matmul and apply a ReLu
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)
                tf.summary.histogram("activations", h_fc)
                return h_fc, output_channel
            else:
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)
                tf.summary.histogram("activations", h_fc)
                return h_fc, output_channel
            
    def dropout_layer(fc_input_layer, keep_prob):
        with tf.name_scope('dropout'):
            """Dropout controls the complexity of the model, prevents co-adaptation of features"""
            print("Using a dropout layer...")
            print("Probability of keeping the layer: {}".format(keep_prob))
            # automatically handles scaling neuron outputs and also masks them
            fc_dropout = tf.nn.dropout(fc_input_layer, keep_prob)
            return fc_dropout
                
    def output_layer(input_channels,num_outputs,input_layer,keep_prob):
            print("\nBuilding the final layer of the network...")
            print("Weight shape: [{},{}], Bias shape: [{}]".\
                  format(input_channels,num_outputs,num_outputs))
            
            W_fc = buildNet.weight_variable([input_channels, num_outputs])
            b_fc = buildNet.bias_variable([num_outputs])
            
            dropout = buildNet.dropout_layer(input_layer,keep_prob)
            y_conv = tf.matmul(dropout, W_fc) + b_fc
            tf.summary.histogram("activations", y_conv)
            return y_conv


