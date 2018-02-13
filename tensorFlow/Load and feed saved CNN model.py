# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:43:03 2018

@author: ellio
"""
# script which loads a saved CNN (will work on any saved model) and feeds data through whilst printing the accuracy but doesnt alter weights
#%% Imports, set directories, seb
name = 'Admin'
funcPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
savePath = 'C:\\Users\\'+name+'\\Desktop\\MLChinese\\Saved script files'
workingPath = 'C:\\Users\\'+name+'\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
LOGDIR = r'C:/Users/'+name+'/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
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
modelPath = '2017-12-15\Chinese_conv_5\Outputs10_LR0.001_Batch128'# path of loaded model relative to LOGDIR
modelName='LR0.001_Iter3590_TestAcc0.8976510167121887.ckpt.meta'
#import modules
import tensorflow as tf
#from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import time as t
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
start = t.time()
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
print("took ",t.time()-start," seconds\n")
#define images and labels as a subset of the data
#this function splits the data and prepares it for use in the network, can be used to loop
#over several numOutputs
def prepareDataSet(numOutputs,trainRatio,CharImages,CharLabels):
    print("preparing dataset")
    start = t.time()
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
    print("took ",t.time()-start," seconds\n")
    return trainData, testLabels, testImages


#%%Build the network

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
   
     
    #%% Open a tensorflow session
    print("Importing graph.....")
    start = t.time()
    sess = tf.InteractiveSession()
    # Initialise all variables
    #tf.global_variables_initializer().run()
    loadLOGDIR = os.path.join(baseLOGDIR,modelPath)
    os.chdir(loadLOGDIR)
    saver = tf.train.import_meta_graph(modelName)
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    graph = tf.get_default_graph()
    print(graph.get_operations())
    x=graph.get_tensor_by_name("images:0")
    y_=graph.get_tensor_by_name("labels:0")
    keep_prob=graph.get_tensor_by_name("dropout/Placeholder:0")
    accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
    print("took ",t.time()-start," seconds\n")
    
    
    
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
            train_accuracy =sess.run(accuracy, \
                         feed_dict={x: batchImages, y_: batchLabels, keep_prob: 1.0})
            #train_writer.add_summary(train_summary, i*trainBatchSize)
            print("train accuracy ",train_accuracy)
        if i % testNum == 0:
            print("Testing the net...")
            test_accuracy = sess.run(accuracy, \
                           feed_dict={x: testImages,y_: oneHot(testLabels,numOutputs), keep_prob: 1.0})
            #test_writer.add_summary(test_summary, i*trainBatchSize)
            saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{}.ckpt".format(learningRate,i,test_accuracy)))
            print("test accuracy ",test_accuracy)
        
            
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
            for trainBatchSize in [2]: 
                iterations = 600*int(len(trainData.labels)/trainBatchSize)
                #LOGDIR, whichTest, numOutputs, learningRate, trainBatchSize, iterations
                neural_net(LOGDIR,whichTest,numOutputs,learning_rate,trainBatchSize,iterations,\
                           trainData,testImages,testLabels,trainRatio)
