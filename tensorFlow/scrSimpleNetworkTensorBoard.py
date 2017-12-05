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

#file Path for functions

user = "Elliot"
#user = "Seb"

funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHub/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
dataPathElliot = 'C:/Users/ellio/Documents/training data/Machine Learning data/'
dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
savePathElliot = 'C:\\Users\\ellio\OneDrive\\Documents\\University\\Year 4\\ML chinese caligraphy\\Graphs'
SebLOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
elliotLOGDIR = r'C:/Users/ellio/Anaconda3/Lib/site-packages/tensorflow/tmp/SimpleNetwork/'

if user == "Elliot":
    funcPath = funcPathElliot
    dataPath = dataPathElliot
    savePath = savePathElliot
    LOGDIR = elliotLOGDIR
else:
    funcPath = funcPathSeb
    dataPath = dataPathSeb
    savePath = savePathSeb
    LOGDIR = SebLOGDIR

whichTest =8
LOGDIR = LOGDIR + str(datetime.date.today()) + '/{}'.format(whichTest)
#make a directory
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

#first part of the next line goes one back in the directory
#os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\\dataHandling')
os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
"""Define the user"""
#funcPath,dataPath,savePath,rootDIR = fF.whichUser('Seb')
#os.chdir("..")

def makeDir(rootDIR,fileName,hparam):
    """Makes a directory automatically to save tensorboard data to"""
    testNum = 0
    LOGDIR = rootDIR + str(datetime.date.today()) + '/' + fileName + '-test-{}'.format(testNum)
    while os.path.exists(LOGDIR):
        testNum += 1
        LOGDIR = rootDIR + str(datetime.date.today()) + '/' + fileName + '-test-{}'.format(testNum)
    #make a directory
    os.makedirs(LOGDIR)
    return LOGDIR


#%%Get the data
#set ration of data to be training and testing
trainRatio = 0.95


print("splitting data...")
startTime=t.time()
#file to open

def subSet(numClasses,images,labels):
      """return subset of characters, i.e. 10 characters with images and labels not 3755"""
      subImages = []
      subLabels = []
      for i in range(len(images)):
          if labels[i] in range(numClasses):
              subImages.append(images[i])
              subLabels.append(labels[i])
      return np.asarray(subImages),np.asarray(subLabels)


#dataPath = savePath
fileName="CharToNumList_10"
labels,images=fF.readNPZ(dataPath,"1001-1100C","saveLabels","saveImages")
nextLabels,nextImages = fF.readNPZ(dataPath,"1101-1200C","saveLabels","saveImages")
labels = np.concatenate((labels,nextLabels),axis=0)
images = np.concatenate((images,nextImages),axis=0)
nextLabels,nextImages = fF.readNPZ(dataPath,"1201-1300C","saveLabels","saveImages")
labels = np.concatenate((labels,nextLabels),axis=0)
images = np.concatenate((images,nextImages),axis=0)
dataLength=len(labels)
#split the data into training and testing
#train data
subImages, subLabels = subSet(2,images,labels)
"""Must convert to numpy array or tensorflow doesn't work!"""
dataLength=len(subLabels)
trainImages = subImages[0:int(dataLength*trainRatio)]
trainLabels = subLabels[0:int(dataLength*trainRatio)]
testImages = subImages[int(dataLength*trainRatio):dataLength]
testLabels = subLabels[int(dataLength*trainRatio):dataLength]

def saveImages(trainImages,trainLabels):
    """Check subset we are using is valid, by matching image to label"""
    os.chdir('C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files\\checkImages')
    for i in range(len(trainImages)):
        tmpImage = Image.fromarray(np.resize(trainImages[i],(40,40)), 'L')
        tmpImage.save('{},label_{}.jpeg'.format(i,trainLabels[i]))

labels = 0;
images = 0;
print("took ",t.time()-startTime," seconds\n")



#%%
print("Building network...")
def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.03), name="W")
    #b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w))
    tf.summary.histogram("weights", w)
    #tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

if len(set(trainLabels)) == len(set(testLabels)):
    numOutputs = len(set(trainLabels))
    print('{} outputs'.format(numOutputs))
else:
    print('\n\nERRR NUMBER OF UNIQUE TEST LABELS DOES NOT MATCH UNIQUE TRAIN LABELS\n\n')
    
numOutputs = 2
inputDim = 40
trainBatchSize = 20

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

#      with tf.name_scope("test_accuracy"):
#        correct_prediction_test = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
#        accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
#        tf.summary.scalar("test_accuracy", accuracy_test)
#    
      merged_summary_op = tf.summary.merge_all()
      embedding = tf.Variable(tf.zeros([len(testLabels), embedding_size]), name="test_embedding")
      assignment = embedding.assign(embedding_input)
      saver = tf.train.Saver()
      
      """Initialise variables, key step, can only make tensorflow objects after this"""
      sess.run(tf.global_variables_initializer())

      """Create the writers"""
      train_writer = tf.summary.FileWriter(os.path.join(LOGDIR,hparam)+'/train')
      train_writer.add_graph(sess.graph)
      test_writer = tf.summary.FileWriter(os.path.join(LOGDIR,hparam)+'/test')
      test_writer.add_graph(sess.graph)
      
      """Embedding for the projector"""
      config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
      embedding_config = config.embeddings.add()
      embedding_config.tensor_name = embedding.name
      embedding_config.sprite.image_path = os.path.join(savePath,'sprite_1024')
      embedding_config.metadata_path = os.path.join(savePath,'spriteLabels.tsv')
      # Specify the width and height of a single thumbnail.
      embedding_config.sprite.single_image_dim.extend([40, 40])
      tf.contrib.tensorboard.plugins.projector.visualize_embeddings(test_writer, config)

      
      print("Creating dataset tensors...")
      tensorCreation = t.time()
      #create dataset for training and validation
      tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
      tr_data = tr_data.shuffle(buffer_size=10000)
      tr_data = tr_data.repeat()
      tr_data = tr_data.batch(trainBatchSize)
      
      val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))
      val_data = val_data.shuffle(buffer_size=10000)
      val_data = val_data.repeat()
      val_data = val_data.batch(len(testLabels))
      #repeat the test dataset infinitely, so that we can loop over its test    
      
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
      
      """Print labels/images/tf.one_hot to check"""
#      print(next_label.eval())
#      print(tf.one_hot(next_label,3373).eval())
#      print(len(tf.one_hot(next_label,3373).eval()))
#      print(next_val_label.eval())
#      print(tf.one_hot(next_val_label,3373).eval())
#      print(len(tf.one_hot(next_val_label,3373).eval()))
      
      epochLength = int(len(trainLabels)/trainBatchSize)
      print('Number of iterations for one epoch: {}'.format(epochLength))
      iterations = 1001
      displayNum = 30
      testNum = 100
      for i in range(iterations): #range 2001
          if i % displayNum == 0:
              print('calculating training accuracy... i={}'.format(i))
              ###debug
#              tempImage = next_image
#              tempLabel = next_label
#              print(tempImage.eval(),len(tempImage.eval()),len(tempImage.eval()[0]))
#              print(tempLabel.eval(),len(tempLabel.eval()))
#              print(tf.one_hot(tempLabel,numOutputs).eval(),len(tf.one_hot(tempLabel,numOutputs).eval()),len(tf.one_hot(tempLabel,numOutputs).eval()[0]))
              ###end debug
              train_accuracy, train_summary = sess.run([accuracy, merged_summary_op], \
                  feed_dict={x: next_image.eval(), \
                             y: tf.one_hot(next_label,numOutputs).eval()})
              train_writer.add_summary(train_summary, i)
          if i % testNum == 0:
              print('calculating test accuracy and saving')
              ###debug
#              tempImage = next_val_image
#              tempLabel = next_val_label
#              print(tempImage.eval(),len(tempImage.eval()),len(tempImage.eval()[0]))
#              print(tempLabel.eval(),len(tempLabel.eval()))
#              print(tf.one_hot(tempLabel,numOutputs).eval(),len(tf.one_hot(tempLabel,numOutputs).eval()),len(tf.one_hot(tempLabel,numOutputs).eval()[0]))
              ###end the debug
              #sess.run(assignment, \
              assign, test_accuracy, test_summary = sess.run([assignment, accuracy, merged_summary_op], \
                       feed_dict={x: next_val_image.eval(),  \
                                  y: tf.one_hot(next_val_label,numOutputs).eval()})
              test_writer.add_summary(test_summary, i)
              saver.save(sess, os.path.join(LOGDIR, "model.ckpt{}".format(learning_rate)), i)              
          if i % epochLength == 0 and i!=0:
              print('Did {} epochs'.format(i/epochLength))
              learning_rate*=0.99
              
          sess.run(train_step, \
                   feed_dict={x: next_image.eval(), \
                              y: tf.one_hot(next_label,numOutputs).eval()})
      train_writer.close()
      test_writer.close()
#          print(sess.run(logits,feed_dict={x: next_image.eval(), \
#                              y: tf.one_hot(next_label,10).eval()}))
    
def make_hparam_string(learning_rate):
  fc_param = "fc=1"
  return "lr_%.0E,%s" % (learning_rate, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-2]:

    # Include "False" as a value to try different model architectures
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
    hparam = make_hparam_string(learning_rate)
    print('Starting run for %s\n' % hparam)
    #LOGDIR = makeDir(rootDIR,fileName,hparam)
	 #Actually run with the new settings
    neural_net(LOGDIR, learning_rate, hparam)

if __name__ == '__main__':
  main()