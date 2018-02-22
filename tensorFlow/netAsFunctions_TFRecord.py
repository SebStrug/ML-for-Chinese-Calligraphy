# -*- coding: utf-8 -*-
"""
Build and train a neural network using TFRecords as input
Look at the bottom of the file for all the inputs
"""

import tensorflow as tf
from tryBuildNet import buildNet
import time
import math #to use radians in rotating the image
import random
from classDataManip import makeDir
import os
from InputTFRecord import inputs

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.
# Initialize `iterator` with training data.


def run_training():
    tf.reset_default_graph()
        
    train_kwargs = {"normalize_images": False, "augment_images": False, "shuffle_data": True}
    train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,\
                                                  train_batch_size,num_epochs,\
                                                  **train_kwargs)
    test_kwargs = {"normalize_images": False, "augment_images": False, "shuffle_data": False}
    test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,\
                                                test_batch_size,0,\
                                                **test_kwargs)

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
    
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""    
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    #Define the placeholders for the images and labels
    # 'None' used to be batch_size << haven't tested None yet
    x = tf.placeholder(tf.float32, [None,inputDim**2], name="images")
    y_ = tf.placeholder(tf.float32, [None,num_output], name="labels")
    
    with tf.name_scope('reshape'):
        # reshape x to a 4D tensor with second and third dimensions being width/height
        x_image = tf.reshape(x, [-1,inputDim,inputDim,1])
    
    tf.summary.image('input', x_image, 4) # Show 4 examples of output images on tensorboard
    
    with tf.name_scope('conv1'):
        """First convolution layer, maps one greyscale image to 32 feature maps"""
        # patch size of 5x5, 1 input channel, 32 output channels (features)
        W_conv1 = weight_variable([5, 5, 1, 32])
        # bias has a component for each output channel (feature)
        b_conv1 = bias_variable([32])
        # convolve x with the weight tensor, add bias and apply ReLU function
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram("activations", h_conv1)
        
    with tf.name_scope('pool1'):
        """Pooling layer, downsamples by 2x"""
        #max pool 2x2 reduces it to 14x14
        h_pool1 = max_pool_2x2(h_conv1)
    
    with tf.name_scope('conv2'):
        """Second convolution layer, maps 32 features maps to 64"""
        # 64 outputs (features) for 32 inputs
        W_conv2 = weight_variable([5, 5, 32, 64])
        # bias has to have an equal number of outputs
        b_conv2 = bias_variable([64])
        # convolve again
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram("activations", h_conv2)
    
    with tf.name_scope('pool2'):
        """Second pooling layer"""
        # pool and reduce to 7x7
        h_pool2 = max_pool_2x2(h_conv2)    
    
#    with tf.name_scope('fc1'):
#        """Fully connected layer, maps features to the number of outputs"""
#        w_fc = weight_variable([inputDim**2,num_output])
#        b_fc = bias_variable([num_output])
#        # calculate the convolution
#        y_conv = tf.matmul(x, w_fc) + b_fc
#        tf.summary.histogram("activations", y_conv)

    with tf.name_scope('fc1'):
        """Fully connected layer 1, after 2 rounds of downsampling, our 28x28 image
        is reduced to 7x7x64 feature maps, map this to 1024 features"""
        # 7*7 image size *64 inputs, fc layer has 1024 neurons
        W_fc1 = weight_variable([12 * 12 * 64, 1024])
        b_fc1 = bias_variable([1024])
        # reshape the pooling layer from 7*7 (*64 inputs) to a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
        # do a matmul and apply a ReLu
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = 'bottleneck')
        tf.summary.histogram("activations", h_fc1)

    with tf.name_scope('dropout'):
        """Dropout controls the complexity of the model, prevents co-adaptation of features"""
        # placeholder for dropout means we can turn it on during training, turn off during testing
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        # automatically handles scaling neuron outputs and also masks them
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    with tf.name_scope('fc2'):
        """Fully connected layer 2, maps 1024 features to the number of outputs"""
        #1024 inputs, 10 outputs
        W_fc2 = weight_variable([1024, num_output])
        b_fc2 = bias_variable([num_output])
        # calculate the convolution
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.summary.histogram("activations", y_conv)
        
    with tf.name_scope("xent"):    
        cross_entropy = tf.reduce_mean(\
                            tf.nn.softmax_cross_entropy_with_logits(\
                                labels=y_, \
                                logits=y_conv))
        tf.summary.scalar("xent",cross_entropy)
        
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # what fraction of bools was correct? Cast to floating point...
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)

    #merge all summaries for tensorboard
    mergedSummaryOp = tf.summary.merge_all()
    # Create a saver to save these summary operations
    saver = tf.train.Saver()
    
    #this is the operation that initialises all the graph variables
    init_op = tf.group(tf.global_variables_initializer(),\
                       tf.local_variables_initializer())        
    
    with tf.Session() as sess:
                
#        tf.global_variables_initializer().run()    
            
        # Create writers
        train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
        train_writer.add_graph(sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/test')
        test_writer.add_graph(sess.graph)
        
        #initialise the variables
        sess.run(init_op)
        
        start_time = time.time()  
        try:
            step = 0
            while True: #train until we run out of epochs
                train_images, train_labels = sess.run([train_image_batch,train_label_batch])
                test_images, test_labels = sess.run([test_image_batch,test_label_batch])      
            
                if step % 30 == 0:
                    train_accuracy, train_summary = sess.run([accuracy, mergedSummaryOp], \
                                 feed_dict={x: train_images, \
                                            y_: tf.one_hot(train_labels,num_output).eval(),\
                                            keep_prob: 1.0})
                    train_writer.add_summary(train_summary, step)
                    print('Step: {}, Training accuracy = {:.3}'.format(step, train_accuracy))
                
                if step % 90 == 0:
                    print("Testing the net...")
                    test_accuracy, test_summary = sess.run([accuracy,mergedSummaryOp], \
                                   feed_dict={x: test_images,\
                                              y_: tf.one_hot(test_labels,num_output).eval(),\
                                              keep_prob: 1.0})
                    test_writer.add_summary(test_summary, step)
                    saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{:.3}.ckpt".\
                                                  format(learning_rate,step,test_accuracy)))
                    print('Step: {}, Test accuracy = {:.3}'.format(step, test_accuracy))
                
                #run the training
                sess.run(train_step, feed_dict={x: train_images,\
                                                y_: tf.one_hot(train_labels,num_output).eval(),\
                                                keep_prob:0.5})              
                step += 1
        except tf.errors.OutOfRangeError:
            duration = time.time() - start_time
            print('Done {} epochs, {} steps, took {:.3} mins.'.\
                  format(num_epochs,step,duration/60))  
        
        train_writer.close()
        test_writer.close()
            
def main():
    run_training()

inputDim = 48
num_output_list = [10]
num_epoch_list = [800]
train_batch_size_list = [128]
learning_rate_list = [1E-3]
test_batch_size = 500

savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved_runs\\'
localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0'

name_of_run = '2conv_2fc'

for num_output in num_output_list:
    train_tfrecord_filename = \
        localPath+'\\train'+str(num_output)+'.tfrecords'
    test_tfrecord_filename = \
        localPath+'\\test'+str(num_output)+'.tfrecords'
    for num_epochs in num_epoch_list:
        for train_batch_size in train_batch_size_list:
            for learning_rate in learning_rate_list:
                LOGDIR = makeDir(savePath,name_of_run,num_output,\
                                     learning_rate,train_batch_size)
                main()
    
    