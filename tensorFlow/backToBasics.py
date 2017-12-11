# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:43:52 2017

@author: Sebastian
Back to basics
"""
#%% Imports, set directories, seb

#funcPath = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
#savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
#workingPath = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow'
#LOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
#%% Imports, set directories, Elliot
funcPath = 'C:\\Users\\ellio\\OneDrive\\Documents\\GitHubPC\\ML-for-Chinese-Calligraphy\\dataHandling'
savePath = 'C:\\Users\\ellio\\Documents\\training data\\Machine learning data'
workingPath = 'C:\\Users\\ellio\\OneDrive\\Documents\\GitHubPC\\ML-for-Chinese-Calligraphy\\tensorFlow'
LOGDIR = r'C:\\Users\\ellio\\Anaconda3\\Lib\\site-packages\\tensorflow\\tmp\\'
#%%

import os
import tensorflow as tf
import numpy as np
import time as t
import datetime
from PIL import Image
import random
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
os.chdir(workingPath)
from classDataManip import subSet

#make a directory to save tensorboard information in 
whichTest = 2
LOGDIR = LOGDIR + str(datetime.date.today()) + '/Chinese_conv_{}/LR1E-3'.format(whichTest)
#make a directory if one does not exist
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

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
#MNIST data
#fileName="MNIST_data"
#MNISTLabels,MNISTImages=fF.readNPZ(savePath,fileName,"saveLabels","saveImages")

#Chinese characters data
CharLabels,CharImages=fF.readNPZ(savePath,"1001to1100","saveLabels","saveImages")
nextLabels,nextImages = fF.readNPZ(savePath,"1101to1200","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
nextLabels,nextImages = fF.readNPZ(savePath,"1201to1300","saveLabels","saveImages")
CharLabels = np.concatenate((CharLabels,nextLabels),axis=0)
CharImages = np.concatenate((CharImages,nextImages),axis=0)
del nextLabels; del nextImages;

#define images and labels as a subset of the data
numOutputs = 10
trainRatio = 0.9
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

#%%Build the network
print("Building the net...")
#Reset the graph (since we are not creating this in a function!)
tf.reset_default_graph()
inputDim = 40
learningRate = 1e-3
trainBatchSize = len(trainLabels)

# function to create weights and biases automatically
# want slightly positive weights/biases for relu to avoid dead nurons
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.03), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        """can't end on a relu!"""
        #act = tf.nn.relu(tf.matmul(input,W) + b)        
        act = tf.matmul(input,W)+b
        tf.summary.histogram("activations", act)
        return act

# Define the placeholders for images and labels
x = tf.placeholder(tf.float32, [None, inputDim**2], name="images")
y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")

with tf.name_scope('reshape'):
    # reshape x to a 4D tensor with second and third dimensions being width/height
    x_image = tf.reshape(x, [-1,inputDim,inputDim,1])

with tf.name_scope('conv1'):
    """First convolution layer, maps one greyscale image to 32 feature maps"""
    # patch size of 5x5, 1 input channel, 32 output channels (features)
    W_conv1 = weight_variable([5, 5, 1, 32])
    # bias has a component for each output channel (feature)
    b_conv1 = bias_variable([32])
    # convolve x with the weight tensor, add bias and apply ReLU function
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
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

with tf.name_scope('pool2'):
    """Second pooling layer"""
    # pool and reduce to 7x7
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    """Fully connected layer 1, after 2 rounds of downsampling, our 28x28 image
    is reduced to 7x7x64 feature maps, map this to 1024 features"""
    # 7*7 image size *64 inputs, fc layer has 1024 neurons
    W_fc1 = weight_variable([10 * 10 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # reshape the pooling layer from 7*7 (*64 inputs) to a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
    # do a matmul and apply a ReLu
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    """Dropout controls the complexity of the model, prevents co-adaptation of features"""
    # placeholder for dropout means we can turn it on during training, turn off during testing
    keep_prob = tf.placeholder(tf.float32)
    # automatically handles scaling neuron outputs and also masks them
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    """Fully connected layer 2, maps 1024 features to the number of outputs"""
    #1024 inputs, 10 outputs
    W_fc2 = weight_variable([1024, numOutputs])
    b_fc2 = bias_variable([numOutputs])
    # calculate the convolution
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Calculate entropy on the raw outputs of y then average across the batch size
with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(\
                        tf.nn.softmax_cross_entropy_with_logits(\
                            labels=y_, \
                            logits=y_conv))
    tf.summary.scalar("xent",cross_entropy)
    
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # what fraction of bools was correct? Cast to floating point...
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy",accuracy)

# Merge all summary operators
mergedSummaryOp = tf.summary.merge_all()
# Create a saver to save these summary operations
saver = tf.train.Saver()

#%%Create the dataset tensors for training and validation data
print("Creating dataset tensors...")
tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
#tr_data = tr_data.shuffle(buffer_size=100,seed=2)
tr_data = tr_data.repeat()
tr_data = tr_data.batch(trainBatchSize)
val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))
#val_data = val_data.shuffle(buffer_size=100)
val_data = val_data.batch(len(testLabels))

#Create the training and validation iterators over batches
tr_iterator = tr_data.make_initializable_iterator()
tr_next_image, tr_next_label = tr_iterator.get_next()
val_iterator = val_data.make_initializable_iterator()
val_next_image, val_next_label = tr_iterator.get_next()
 
#%% Open a tensorflow session
print("Initialising the net...")
sess = tf.InteractiveSession()
# Initialise all variables
tf.global_variables_initializer().run()
# Initailise the iterator
sess.run(tr_iterator.initializer)

# Create a writer
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
train_writer.add_graph(sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/test')
test_writer.add_graph(sess.graph)

#%% Start training!
print("Starting training!")
epochLength = int(len(trainLabels)/trainBatchSize) #no. of iterations per epoch
whichEpoch = 0
print("Number of iterations per epoch: {}".format(epochLength))
iterations = 200
displayNum = 2
testNum = 10
for i in range(iterations):
    """Check a random value in the batch matches its label"""
    batchImages, batchLabels = tr_next_image, tr_next_label
#    randomIndex = random.randint(0,trainBatchSize-1)
#    print(display(batchImages.eval()[randomIndex], inputDim),batchLabels.eval()[randomIndex])
    
    if i % displayNum == 0:
        train_accuracy, train_summary =sess.run([accuracy, mergedSummaryOp], \
                     feed_dict={x: batchImages.eval(), y_: tf.one_hot(batchLabels,numOutputs).eval(),keep_prob: 1.0})
        train_writer.add_summary(train_summary, i)
        
    if i % testNum ==0:
        print("Testing the net...")
        testBatchImages, testBatchLabels = val_next_image, val_next_label
        test_accuracy, test_summary = sess.run([accuracy,mergedSummaryOp], \
                       feed_dict={x: testBatchImages.eval(),y_: tf.one_hot(testBatchLabels,numOutputs).eval(),keep_prob: 1.0})
        test_writer.add_summary(test_summary, i)
        saver.save(sess, os.path.join(LOGDIR, "model.ckpt{}".format(learningRate)), i)
        
    if i % epochLength == 0 and i != 0:
        whichEpoch += 1
        print("Did {} epochs".format(whichEpoch))
    # run a training step   
    sess.run(train_step, \
             feed_dict={x: batchImages.eval(), \
                        y_: tf.one_hot(batchLabels,numOutputs).eval(), \
                        keep_prob: 0.5})
train_writer.close()

#%% Test trained model
#print("Testing the net...")
#testBatchImages, testBatchLabels = val_next_image, val_next_label
#print(sess.run(accuracy, \
#               feed_dict={x: testBatchImages.eval(), \
#                          y_: tf.one_hot(testBatchLabels,numOutputs).eval(), \
#                          keep_prob: 1.0}))

