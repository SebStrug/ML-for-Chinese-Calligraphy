# -*- coding: utf-8 -*-
"""
Build and train a neural network using TFRecords as input
Look at the bottom of the file for all the inputs
"""

import tensorflow as tf
from tryBuildNet import buildNet
import itertools
import time
import math #to use radians in rotating the image
import numpy as np
import matplotlib.pyplot as plt
import random
from classDataManip import makeDir
import os
import scipy as sp
from InputTFRecord import inputs
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.
# Initialize `iterator` with training data.

#%%Func defs for visualising filters
def removeAndSwapAxes(array):
    foo=np.swapaxes(array,2,3)
    boo=np.swapaxes(foo,1,2)
    return boo
    
def show_activations(features,featureDim, num_out, path, name, show = False, save = False):
    test_images = features

    size_figure_grid = int(num_out/8)
    fig, ax = plt.subplots(8, size_figure_grid, figsize=(8, int(8*8/size_figure_grid)))
    print("Number of images: {}, size of grid: {}".format(num_out, size_figure_grid))
    for i, j in itertools.product(range(8), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(num_out):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (featureDim, featureDim)) )#, cmap='gray')
    #label = 'Epoch {0}'.format(num_epoch)
    #fig.text(0.5, 0.04, label, ha='center')
    if save:
        os.chdir(path)
        plt.savefig(name)
    if show:
        plt.show()
    else:
        plt.close() 
        
def maximum_activation(layerActivations):
    layerHighest = []
    activationTot = np.sum(layerActivations,axis=(2,3))
    maxIndices = np.argmax(activationTot,axis=0)
    for i in range(0,layerActivations.shape[1]):
            layerHighest.append(layerActivations[int(maxIndices[i])][i])
    return layerHighest
        
def convert_weight_filters(layerWeights):
    weight_filters = [layerWeights[:,:,:,i] for i in range(layerWeights.shape[3])]
    weight_filters = [np.sum(i,axis=2) for i in weight_filters]
    weight_upscaled = [sp.misc.imresize(weight_filters[i],10.0) for i \
                        in range(layerWeights.shape[3])]
    return weight_upscaled

#%%
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
#        buildNet.conv_layer('conv_3',conv_layer_2, output_dim, 4, [1,1], \
#                            output_channels, 96, do_pool=False)
#    conv_layer_4, output_dim, output_channels = \
#        buildNet.conv_layer('conv_4',conv_layer_3, output_dim, 3, [1,1], \
#                            output_channels, 128, do_pool=False)
#    conv_layer_5, output_dim, output_channels = \
#        buildNet.conv_layer('conv_5',conv_layer_4, output_dim, 2, [1,1], \
#                        output_channels, 160, do_pool=True)
#    conv_layer_6, output_dim, output_channels = \
#        buildNet.conv_layer('conv_6',conv_layer_5, output_dim, 2, [1,1], \
#                        output_channels, 192, do_pool=False)
    fc_layer_1, output_channels = \
        buildNet.fc_layer('fc_1', conv_layer_2, output_dim, output_channels, \
                          1024, do_pool=True)
    y_conv = buildNet.output_layer(output_channels, num_output, fc_layer_1,0.5)
    
    """Simple network"""
#    y_conv = buildNet.output_layer(inputDim**2, num_output, x, 1)      
      
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
    
    print("Starting session...")
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        conv1Activations = graph.get_tensor_by_name("conv_1/Relu:0")
        conv1Weights = graph.get_tensor_by_name("conv_1/Variable:0")
        
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
        
#        #visualise conv_1 features
#        with tf.variable_scope('conv_1') as scope_conv:
#            weights = tf.get_variable('weights')
#            # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
#            x_min = tf.reduce_min(weights)
#            x_max = tf.reduce_max(weights)
#            weights_0_to_1 = (weights - x_min) / (x_max - x_min)
#            weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
#            # to tf.image_summary format [batch_size, height, width, channels]
#            weights_transposed = tf.transpose (weights_0_to_255_uint8, [train_batch_size, 48, 48, 32])
#            # this will display random 3 filters from the 64 in conv1
#            tf.image_summary('conv_1/weights', weights_transposed, max_images=10)
        
        try:
            print("Starting training...")
            step = 0
            while True: #train until we run out of epochs
                train_images, train_labels = sess.run([train_image_batch,train_label_batch])
                test_images, test_labels = sess.run([test_image_batch,test_label_batch])      
                
                if step == 0:
                    image_for_visualisation = [train_images[0]]
                
                if step % 30 == 0:
                    train_accuracy, train_summary = sess.run([accuracy, mergedSummaryOp], \
                                 feed_dict={x: train_images, \
                                            y_: tf.one_hot(train_labels,num_output).eval(),\
                                            keep_prob: 1.0})
                    train_writer.add_summary(train_summary, step)
                    print('Step: {}, Training accuracy = {:.3}'.format(step, train_accuracy))
                    
                    # Visualise the activations for this one image
                    layer1Activations=sess.run(conv1Activations,feed_dict={x: image_for_visualisation, keep_prob: 1.0})
                    layer1Activations=removeAndSwapAxes(layer1Activations)
                    layerHighest = maximum_activation(layer1Activations)
                    show_activations(layerHighest, len(layerHighest[0]), len(layerHighest), \
                                     "C:\\Users\\Sebastian\\Desktop\\MLChinese\\Visualising_filters\\100Out_Visualise_whileTraining",\
                                     "layer1Features_step{}.jpg".format(step), show=False, save=True)
                    # Visualise the weights
                    layer1Weights=sess.run(conv1Weights)
                    weights_upscaled = convert_weight_filters(layer1Weights)
                    show_activations(weights_upscaled, len(weights_upscaled[0]), len(weights_upscaled),\
                                     "C:\\Users\\Sebastian\\Desktop\\MLChinese\\Visualising_filters\\100Out_Visualise_whileTraining", \
                                     'layer1Weights_step{}.jpg'.format(step),\
                                     show=False, save=True)
                    
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
num_output_list = [100]
num_epoch_list = [300]
train_batch_size_list = [128]
learning_rate_list = [1E-3]
test_batch_size = 500

#dataPath, LOGDIR, rawDataPath = fF.whichUser("Elliot")
#savePath=LOGDIR
#localPath=os.path.join(dataPath,"Machine learning data/TFrecord")

savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved_runs\\'
localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0'


name_of_run = 'Generating_activations'

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
    
    