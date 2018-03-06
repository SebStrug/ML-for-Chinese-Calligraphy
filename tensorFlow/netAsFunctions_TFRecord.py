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
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.
# Initialize `iterator` with training data.


def run_training():
    tf.reset_default_graph()
        
    train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data": True}
    train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,\
                                                  train_batch_size,num_epochs,\
                                                  **train_kwargs)
    test_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data": True}
    test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,\
                                                test_batch_size,0,\
                                                **test_kwargs)
    
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
    
    conv_layer_1, output_dim, output_channels = \
        buildNet.conv_layer('conv_1', x_image, inputDim, 5, [1,1], 1, 32, do_pool=True)
    conv_layer_2, output_dim, output_channels = \
        buildNet.conv_layer('conv_2', conv_layer_1, output_dim, 5, [1,1], \
                            output_channels, 64, do_pool=True)
#    conv_layer_3, output_dim, output_channels = \
#        buildNet.conv_layer('conv_3',conv_layer_2, output_dim, 4, \
#                            output_channels, 96, do_pool=False)
#    conv_layer_4, output_dim, output_channels = \
#        buildNet.conv_layer('conv_4',conv_layer_3, output_dim, 3, \
#                            output_channels, 128, do_pool=False)
#    conv_layer_5, output_dim, output_channels = \
#        buildNet.conv_layer('conv_5',conv_layer_4, output_dim, 2, \
#                        output_channels, 160, do_pool=True)
#    conv_layer_6, output_dim, output_channels = \
#        buildNet.conv_layer('conv_6',conv_layer_5, output_dim, 2, \
#                        output_channels, 256, do_pool=False)
#    fc_layer_1, output_channels = \
#        buildNet.fc_layer('fc_1', conv_layer_2, output_dim, output_channels, \
#                          1024, do_pool=True)
#    y_conv = buildNet.output_layer(output_channels, num_output, fc_layer_1,0.5)
    
    """Simple network"""
    y_conv = buildNet.output_layer(inputDim**2, num_output, x, 1)      
      
    with tf.name_scope("xent"):    
        cross_entropy = tf.reduce_mean(\
                            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv))
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
        maxAccuracy=0
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
                    if test_accuracy > maxAccuracy:
                        maxAccuracy=test_accuracy
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
            

inputDim = 48
num_output_list = [10]
num_epoch_list = [10]
train_batch_size_list = [128]
learning_rate_list = [1E-3]
test_batch_size = 500

#dataPath, LOGDIR, rawDataPath = fF.whichUser("Elliot")
#savePath=LOGDIR
#localPath=os.path.join(dataPath,"Machine learning data/TFrecord")

savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved_runs\\'
localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0'


name_of_run = 'testing_code'

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
                run_training()
    
    