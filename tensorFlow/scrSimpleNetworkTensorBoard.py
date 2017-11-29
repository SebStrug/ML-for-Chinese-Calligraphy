# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:25:07 2017

@author: ellio
"""
import os
import tensorflow as tf
import numpy as np
import time as t
import datetime
from PIL import Image

#%%Notes
""" .eval() converts a tensor within a session into its real output.
    so tf.one_hot(X).eval() turns the one_hot tensor into a numpy array as needed,
    before this we need to do 'with sess.as_default()' so all the variables are run
    in the same session
    
    """

#%%Load Data
#first part of the next line goes one back in the directory
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\\dataHandling')
from classFileFunctions import fileFunc as fF 
"""Define the user"""
funcPath,dataPath,savePath,rootDIR = fF.whichUser('Seb')
os.chdir("..")

def makeDir(rootDIR,hparam):
    """Makes a directory automatically to save tensorboard data to"""
    testNum = 0
    LOGDIR = rootDIR + str(datetime.date.today()) + '/test-{}'.format(testNum)
    while os.path.exists(LOGDIR):
        testNum += 1
        LOGDIR = rootDIR + str(datetime.date.today()) + '/test-{}'.format(testNum)
    #make a directory
    os.makedirs(LOGDIR)
    return LOGDIR

#%%Get the data
#set ration of data to be training and testing
trainRatio = 0.95

print("splitting data...")
startTime=t.time()
#file to open

dataPath = savePath

fileName="1001to1100"
labels,images=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")
dataLength=len(labels)
#split the data into training and testing
#train data
trainImages = images[0:int(dataLength*trainRatio)]
trainLabels = labels[0:int(dataLength*trainRatio)]
testImages = images[int(dataLength*trainRatio):dataLength]
testLabels = labels[int(dataLength*trainRatio):dataLength]
labels = 0;
images = 0;
print("took ",t.time()-startTime," seconds\n")
 
#%%
def createSpriteLabels(images,labels):
    """Create a 32x32 image of sprites of the characters"""
    spriteImages = images[0:32*32] #array form
    convSpriteImage = [Image.fromarray(np.resize(i,(40,40)), 'L') for i in spriteImages] #convert to image
    dimensions = 40*32
    montage = Image.new(mode='RGBA', size=(dimensions, dimensions), color=(0,0,0,0))
    offset_x = offset_y = 0
    row_size = 32
    i = 0
    for image in convSpriteImage:
        montage.paste(image, (offset_x, offset_y))
        if i % row_size == row_size-1: 
            offset_y += 40
            offset_x = 0
        else:
            offset_x += 40
        i += 1
    montage.save(savePath + '/sprite_{}'.format(32**2), "png")
    
    spriteLabels = labels[0:32*32]
    with open(savePath + "/spriteLabels.tsv", "w") as record_file:
        record_file.write("Character\n")
        for i in spriteLabels:
            record_file.write('{}\n'.format(i))
    return montage, record_file


#%% Check images are correct
#image0 = Image.fromarray(np.resize(trainImages[0],(40,40)), 'L')
#label0 = trainLabels[0]
#image3755 = Image.fromarray(np.resize(trainImages[3755],(40,40)), 'L')
#label3755 = trainLabels[3755]
#print(image0,label0)
#print(image3755,label3755)
#
#for i in range(len(trainLabels)):
#    if trainLabels[i] == 2604:
#        print(i)
    

#%%
print("Building network...")
def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

if len(set(trainLabels)) == len(set(testLabels)):
    numOutputs = len(set(trainLabels))
    print('{} outputs'.format(numOutputs))
else:
    print('\n\nERRR NUMBER OF UNIQUE TEST LABELS DOES NOT MATCH UNIQUE TRAIN LABELS\n\n')
    
#numOutputs = 3755
inputDim = 40

def neural_net(LOGDIR, learning_rate, hparam):
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  with sess.as_default():
      # Setup placeholders, and reshape the data
      x = tf.placeholder(tf.float32, shape=[None, pow(inputDim,2)], name="x")
      x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
      tf.summary.image('input', x_image, 3)
      y = tf.placeholder(tf.float32, shape=[None,numOutputs], name="labels")
      
      """With conv layer"""
#      conv1 = conv_layer(x_image, 1, 64, "conv")
#      #the next line pools it twice to keep it simple, reduce computational complexity
#      conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#      flattened = tf.reshape(conv_out, [-1, 10 * 10 * 64])  #10*10 or 20*20
#      embedding_input = flattened
#      embedding_size = 10*10*64
#      logits = fc_layer(flattened, 10*10*64, 3373, "fc")
      
      """Without conv layer"""
      embedding_input = x
      embedding_size = pow(inputDim,2)
      logits = fc_layer(x, pow(inputDim,2), numOutputs, "fc")
    
      with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)
    
      with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(xent)
    
      with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    
      summ = tf.summary.merge_all()
    
      embedding = tf.Variable(tf.zeros([1000, embedding_size]), name="test_embedding")
      assignment = embedding.assign(embedding_input)
      saver = tf.train.Saver()
      
      """Initialise variables, key step, can only make tensorflow objects after this"""
      sess.run(tf.global_variables_initializer())
      
      """Create the writers"""
      train_writer = tf.summary.FileWriter(os.path.join(LOGDIR,hparam)+'/train')
      train_writer.add_graph(sess.graph)
      test_writer = tf.summary.FileWriter(os.path.join(LOGDIR,hparam)+'/test')
      test_writer.add_graph(sess.graph)
      
      print("Creating dataset tensors...")
      tensorCreation = t.time()
      #create dataset for training and validation
      tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
      tr_data = tr_data.repeat()
      #take a batch of 128
      tr_data = tr_data.batch(128)
      val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))
      val_data = val_data.shuffle(buffer_size=10000)
      #repeat the test dataset infinitely, so that we can loop over its test
      val_data = val_data.repeat()
      val_data  = val_data.batch(1000)
    
      print("took {} seconds\n".format(t.time()-tensorCreation))
      
      # create TensorFlow Iterator object
      iteratorCreation = t.time()
      print("Creating the iterator...")
      #training iterator
      tr_iterator = tr_data.make_initializable_iterator()
      next_image, next_label = tr_iterator.get_next()
      #validation iterator (not really iterator, takes in all test values)
      val_iterator = val_data.make_initializable_iterator()
      next_val_image, next_val_label = val_iterator.get_next()
      print("took {} seconds\n".format(t.time()-iteratorCreation))
      
      print("Initialising the iterator...")
      iteratorInitialisation = t.time()
      sess.run(tr_iterator.initializer)
      sess.run(val_iterator.initializer)  
      print("took {} seconds\n".format(t.time()-iteratorInitialisation))
      
      print(tf.one_hot(next_label,3755).eval())
      print(len(tf.one_hot(next_label,3755).eval()))
      
      numEpochs = 0
      print('Number of iterations for one epoch: {}'.format(len(trainLabels)/128))
      for i in range(300001): #range 2001
          if i % 30 == 0:
              print('calculating training accuracy... i={}'.format(i))
              [train_accuracy, s] = sess.run([accuracy, summ], \
                  feed_dict={x: next_image.eval(), \
                             y: tf.one_hot(next_label,3755).eval()})
              train_writer.add_summary(s, i)
          if i % 90 == 0:
              print('did 500, saving')
              [assign, test_accuracy, s] = sess.run([assignment, accuracy, summ], \
                       feed_dict={x: next_val_image.eval()[:1000],  \
                                  y: tf.one_hot(next_val_label,3755).eval()[:1000]})
              test_writer.add_summary(s, i)
              saver.save(sess, os.path.join(LOGDIR, "model.ckpt{}".format(learning_rate)), i)
          if i % (len(trainLabels)/128) == 0:
              numEpochs += 1
              print('Did {} epochs'.format(numEpochs))
          sess.run(train_step, \
                   feed_dict={x: next_image.eval(), \
                              y: tf.one_hot(next_label,3755).eval()})
    
def make_hparam_string(learning_rate):
  fc_param = "fc=1"
  return "lr_%.0E,%s" % (learning_rate, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [5E-4]:

    # Include "False" as a value to try different model architectures
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
    hparam = make_hparam_string(learning_rate)
    print('Starting run for %s\n' % hparam)
    LOGDIR = makeDir(rootDIR,hparam)
	 #Actually run with the new settings
    neural_net(LOGDIR, learning_rate, hparam)


if __name__ == '__main__':
  main()