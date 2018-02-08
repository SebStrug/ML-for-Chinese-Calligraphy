# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:43:03 2018

@author: ellio
"""

#%% Imports, set directories, seb
name = 'Admin'
funcPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
savePath = 'C:\\Users\\'+name+'\\Desktop\\MLChinese\\Saved script files'
workingPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
LOGDIR = r'C:/Users/'+name+'/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
#%% Imports, set directories, Elliot
#funcPath = 'C:\\Users\\ellio\\OneDrive\\Documents\\GitHubPC\\ML-for-Chinese-Calligraphy\\dataHandling'
#savePath = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'
#workingPath = 'C:\\Users\\ellio\\OneDrive\\Documents\\GitHubPC\\ML-for-Chinese-Calligraphy\\tensorFlow'
#LOGDIR = r'C:\\Users\\ellio\\Anaconda3\\Lib\\site-packages\\tensorflow\\tmp\\'
#%%

import os
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from classDataManip import subSet,oneHot,makeDir,Data,createSpriteLabels
#set paths
dataPath, LOGDIR = fF.whichUser("Elliot")
#import modules
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
#from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
#import time as t
#from PIL import Image



#%%Display a MNIST image
def display(img, inputDim, threshold=200):
    """Run as print(display(image_array))"""
    render = ''
    for i in range(len(img)):
        if i % inputDim == 0:
            render += '\n'
        if img[i] > threshold:
            render += '@'
        else:
            render += '.'
    return render
                
#%%Import the data
print("Importing the data...")
inputDim = 40
#MNIST data
#fileName="MNIST_data"
#MNISTLabels,MNISTImages=fF.readNPZ(savePath,fileName,"saveLabels","saveImages")

#Chinese characters data
CharLabels,CharImages=fF.readNPZ(dataPath,"1001to1100","saveLabels","saveImages")
nextLabels,nextImages = fF.readNPZ(dataPath,"1101to1200","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
nextLabels,nextImages = fF.readNPZ(dataPath,"1201to1300","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
del nextLabels; del nextImages;

#define images and labels as a subset of the data
#this function splits the data and prepares it for use in the network, can be used to loop
#over several numOutputs
def prepareDataSet(numOutputs,trainRatio,CharImages,CharLabels):

    images, labels = subSet(numOutputs,CharImages,CharLabels)
    dataLength = len(labels) #how many labels/images do we have?
    #del MNISTLabels; del MNISTImages; 
    del CharLabels; del CharImages; #free up memory
    
    #define the training and testing images
    trainImages = images[0:int(dataLength*trainRatio)]
    trainLabels = labels[0:int(dataLength*trainRatio)]
    testImages = images[int(dataLength*trainRatio):dataLength]
    testLabels = labels[int(dataLength*trainRatio):dataLength]
    del images; del labels;        
    
    trainData = Data(trainImages,trainLabels)
    del trainImages,trainLabels
    #%% Create sprites and labels for the embedding
    #os.chdir(workingPath)
    #from classDataManip import createSpriteLabels
    # How many sprites do we want to create (must be a square) > last value in function
    montage, record_file = createSpriteLabels(testImages,testLabels,1024,dataPath)
    del montage; del record_file #don't need to save these, waste space
    return trainData, testLabels, testImages


#%%Build the network
print("Building the net...")
#Reset the graph (since we are not creating this in a function!)


def neural_net(baseLOGDIR,whichTest,numOutputs,learningRate,trainBatchSize,\
               iterations,trainData,testImages,testLabels,trainRatio):
    tf.reset_default_graph()
    LOGDIR = makeDir(baseLOGDIR,whichTest,numOutputs,learningRate,trainBatchSize,trainRatio)
    # function to create weights and biases automatically
    # want slightly positive weights/biases for relu to avoid dead nurons
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
    def load_variable(graph,name):
        return graph.get_tensor_by_name(name)
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""    
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    #sess=tf.Session()    
    #First let's load meta graph and restore weights
#    print("Loading saved net")
#    saver = tf.train.import_meta_graph(os.path.join(baseLOGDIR,'2017-12-15/Chinese_conv_5/Outputs10_LR0.001_Batch128/LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta'))
#    chkp.print_tensors_in_checkpoint_file("LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta", tensor_name='', all_tensors=True)
#    saver.restore(sess,os.path.join(baseLOGDIR,'2017-12-15/Chinese_conv_5/Outputs10_LR0.001_Batch128/LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta'))
#    graph = tf.get_default_graph()
#    print("graph:",graph)
    
#    # Define the placeholders for images and labels
#    print("create and assign variables")
#    x = tf.placeholder(tf.float32, [None, inputDim**2], name="images")
#    x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
#    # Show 4 examples of output images
#    tf.summary.image('input', x_image, 4)
#    y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")
    
#    with tf.name_scope('reshape'):
#        # reshape x to a 4D tensor with second and third dimensions being width/height
#        x_image = tf.reshape(x, [-1,inputDim,inputDim,1])
#    
#    with tf.name_scope('conv1'):
#        """First convolution layer, maps one greyscale image to 32 feature maps"""
#        # patch size of 5x5, 1 input channel, 32 output channels (features)
#        W_conv1 = load_variable(graph,"W_conv1:0")
#        # bias has a component for each output channel (feature)
#        b_conv1 = load_variable(graph,"b_conv1:0")
#        # convolve x with the weight tensor, add bias and apply ReLU function
#        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#        tf.summary.histogram("activations", h_conv1)
#        
#    with tf.name_scope('pool1'):
#        """Pooling layer, downsamples by 2x"""
#        #max pool 2x2 reduces it to 14x14
#        h_pool1 = max_pool_2x2(h_conv1)
#    
#    with tf.name_scope('conv2'):
#        """Second convolution layer, maps 32 features maps to 64"""
#        # 64 outputs (features) for 32 inputs
#        W_conv2 = load_variable(graph,"W_conv2:0")
#        # bias has to have an equal number of outputs
#        b_conv2 = load_variable(graph,"b_conv2:0")
#        # convolve again
#        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#        tf.summary.histogram("activations", h_conv2)
#    
#    with tf.name_scope('pool2'):
#        """Second pooling layer"""
#        # pool and reduce to 7x7
#        h_pool2 = max_pool_2x2(h_conv2)
#    
#    with tf.name_scope('fc1'):
#        """Fully connected layer 1, after 2 rounds of downsampling, our 28x28 image
#        is reduced to 7x7x64 feature maps, map this to 1024 features"""
#        # 7*7 image size *64 inputs, fc layer has 1024 neurons
#        W_fc1 = load_variable(graph,"W_fc1:0")
#        b_fc1 = load_variable(graph,"b_fc1:0")
#        # reshape the pooling layer from 7*7 (*64 inputs) to a batch of vectors
#        h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
#        # do a matmul and apply a ReLu
#        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#        tf.summary.histogram("activations", h_fc1)
#    
#    with tf.name_scope('dropout'):
#        """Dropout controls the complexity of the model, prevents co-adaptation of features"""
#        # placeholder for dropout means we can turn it on during training, turn off during testing
#        keep_prob = tf.placeholder(tf.float32)
#        # automatically handles scaling neuron outputs and also masks them
#        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#    
#    with tf.name_scope('fc2'):
#        """Fully connected layer 2, maps 1024 features to the number of outputs"""
#        #1024 inputs, 10 outputs
#        W_fc2 = load_variable(graph,"W_fc2:0")
#        b_fc2 = load_variable(graph,"b_fc2:0")
#        # calculate the convolution
#        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#        tf.summary.histogram("activations", y_conv)
#    
#    
#    # Calculate entropy on the raw outputs of y then average across the batch size
#    with tf.name_scope("xent"):
#        cross_entropy = tf.reduce_mean(\
#                            tf.nn.softmax_cross_entropy_with_logits(\
#                                labels=y_, \
#                                logits=y_conv))
#        tf.summary.scalar("xent",cross_entropy)
#        
#    with tf.name_scope("train"):
#        train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
#    
#    with tf.name_scope("accuracy"):
#        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#        # what fraction of bools was correct? Cast to floating point...
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        tf.summary.scalar("accuracy",accuracy)
#    
#    # Merge all summary operators
#    mergedSummaryOp = tf.summary.merge_all()
#    # Embedding variables for the projector
#    embedding_var = tf.Variable(tf.zeros([len(testLabels), 1024]), name="test_embedding")
#    #assignment = embedding_var.assign(h_fc1)
#    # Create a saver to save these summary operations AND the embedding
#    saver = tf.train.Saver()
    
  
     
    #%% Open a tensorflow session
    print("Importing graph.....")
    sess = tf.InteractiveSession()
    # Initialise all variables
    #tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph(os.path.join(baseLOGDIR,'2017-12-15/Chinese_conv_5/Outputs10_LR0.001_Batch128/LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta'))
    #saver.restore(sess,tf.train.latest_checkpoint('./'))#os.path.join(baseLOGDIR,'2017-12-15/Chinese_conv_5/Outputs10_LR0.001_Batch128/LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta'))
    
    graph = tf.get_default_graph()
    print(graph.get_operations())
    x=graph.get_tensor_by_name("images:0")
    y_=graph.get_tensor_by_name("labels:0")
    keep_prob=graph.get_tensor_by_name("dropout/Placeholder:0")
    accuracy=graph.get_tensor_by_name("accuracy/accuracy:0")
    
    
    
    
    # Create writers
    train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/test')
    test_writer.add_graph(sess.graph)
#    embedding_writer = tf.summary.FileWriter(LOGDIR)
#    embedding_writer.add_graph(sess.graph)
#    
#    #%% Embedding for the projector
#    """To have the embedding save properly, we need to initialise its own summary,
#    which needs to write just to the base LOGDIR, otherwise we will have problems loading it"""
#    # embedding for the projection of higher dimensional space
#    config = projector.ProjectorConfig()
#    # Can have multiple embeddings, this only adds one
#    embedding = config.embeddings.add()
#    embedding.tensor_name = embedding_var.name
#    # Link the tensor to the label and sprite path
#    embedding.sprite.image_path = os.path.join(savePath,'spriteImages')
#    embedding.metadata_path = os.path.join(savePath,'spriteLabels.tsv')
#    # Specify the width and height of a single thumbnail.
#    embedding.sprite.single_image_dim.extend([inputDim,inputDim])
#    projector.visualize_embeddings(embedding_writer, config)
#    
    
    #%% Start training!
    print("Starting training!")
    trainData.imagePos=0
    trainData.labelPos=0
    epochLength = int(len(trainData.labels)/trainBatchSize) #no. of iterations per epoch
    whichEpoch = 0
    print("Number of iterations per epoch: {}".format(epochLength))
    displayNum = 30
    testNum = 200
    for i in range(iterations):
        """Check a random value in the batch matches its label"""
        batchImages, batchLabels = trainData.nextImageBatch(trainBatchSize), trainData.nextOneHotLabelBatch(trainBatchSize,numOutputs)

    #    randomIndex = random.randint(0,trainBatchSize-1)
    #    print(display(batchImages.eval()[randomIndex], inputDim),batchLabels.eval()[randomIndex])
        
        if i % displayNum == 0:
            train_accuracy, train_summary =sess.run(accuracy, \
                         feed_dict={x: batchImages, y_: batchLabels, keep_prob: 1.0})
            train_writer.add_summary(train_summary, i*trainBatchSize)
            
        if i % testNum == 0:
            print("Testing the net...")
            test_accuracy, test_summary = sess.run(accuracy, \
                           feed_dict={x: testImages,y_: oneHot(testLabels,numOutputs), keep_prob: 1.0})
            test_writer.add_summary(test_summary, i*trainBatchSize)
            saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{}.ckpt".format(learningRate,i,test_accuracy)))
            #summary and assignment for the embedding
#            assign, embedding_summary = sess.run([assignment,mergedSummaryOp], \
#                        #complex powers so that it matches up with number of sprites generated
#                         feed_dict={x: testImages[:1024],y_: oneHot(testLabels,numOutputs)[:1024],keep_prob: 1.0})
#            embedding_writer.add_summary(embedding_summary,i*trainBatchSize)
            
        if i % epochLength == 0 and i != 0:
            whichEpoch += 1
            print("Did {} epochs".format(whichEpoch))
        # run a training step   
#        sess.run(train_step,feed_dict={x: batchImages,y_: batchLabels, keep_prob: 0.5})
    train_writer.close()
    test_writer.close()


#%% Run model function multiple times
whichTest = 1
#trainRatio = 0.8
for numOutputs in [10]:
    for trainRatio in [0.8]:
        trainData,testLabels,testImages = prepareDataSet(numOutputs,trainRatio,CharImages,CharLabels)
        for learning_rate in [1E-4]:
            for trainBatchSize in [128]: 
                iterations = 600*int(len(trainData.labels)/trainBatchSize)
                #LOGDIR, whichTest, numOutputs, learningRate, trainBatchSize, iterations
                neural_net(LOGDIR,whichTest,numOutputs,learning_rate,trainBatchSize,iterations,\
                           trainData,testImages,testLabels,trainRatio)
