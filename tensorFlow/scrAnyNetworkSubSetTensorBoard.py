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

funcPathElliot = 'C:/Users/ellio/OneDrive/Documents/GitHubPC/ML-for-Chinese-Calligraphy/dataHandling'
funcPathSeb = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling'
dataPathElliot = 'C:/Users/ellio/Documents/training data/Machine Learning data/'
dataPathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\Converted\\All C Files'
savePathSeb = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved script files'
savePathElliot = 'C:\\Users\\ellio\OneDrive\\Documents\\University\\Year 4\\ML chinese caligraphy\\Graphs\\'
SebLOGDIR = r'C:/Users/Sebastian/Anaconda3/Lib/site-packages/tensorflow/tmp/ChineseCaligCNN/'
elliotLOGDIR = r'C:/Users/ellio/Anaconda3/Lib/site-packages/tensorflow/tmp/'

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


whichTest = 1
LOGDIR = LOGDIR + str(datetime.date.today()) + '10File/test{}'.format(whichTest)
#make a directory
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
from classImageFunctions import imageFunc as im
os.chdir("..")
#%% set Data size Parameters
numConvOutputs1 = 32
numConvOutputs2= 64
numFcOutputs1 = 1024
numOutputs = 10
inputDim = 40

#%%
def subSet(numClasses,images,labels):
    labelsToExtract = list(range(numClasses))#list of labels that we want to extract
    indicesToExtract = []#the indices that we want to extract from the original data
    for i in range(len(labels)):
        if labels[i] in labelsToExtract:
            indicesToExtract.append(i)
    subSetLabels=np.zeros(len(indicesToExtract))  
    subSetImages=np.zeros((len(indicesToExtract),inputDim*inputDim))
    j=0
    for i in indicesToExtract:
            subSetLabels[j]=labels[i]
            subSetImages[j]=images[i]
            j+=1
    return subSetLabels.astype(int),subSetImages.astype(int)
        
            
        
    
#%%Get the data
#set ration of data to be training and testing

trainRatio = 0.90


print("splitting data...")
startTime=t.time()
#file to open
fileName="1001-1100C"
fullLabels,fullImages=fF.readNPZ(dataPath,fileName,"saveLabels","saveImages")
#############################################################
#if taking a subset of the data
labels,images = subSet(10,fullImages,fullLabels)
#if taking all data
#labels,images = fullLabels, fullImages
fullImages=0
fullLabels = 0
#############################################################
dataLength=len(labels)
#split the data into training and testing
#train data
trainImages = images[0:int(dataLength*trainRatio)]
trainLabels = labels[0:int(dataLength*trainRatio)]
testImages = images[int(dataLength*trainRatio):dataLength]
testLabels = labels[int(dataLength*trainRatio):dataLength]
# uncomment to take a small sample instead 
#trainImages = images[0:3000]
#trainLabels = labels[0:3000]
#testImages = images[0:3000]
#testLabels = labels[0:3000]
trainLength = len(trainLabels)
testLength = len(testLabels)
labels = 0;
images = 0;
print("took ",t.time()-startTime," seconds\n")



#%%
print("Building network...")
def conv_layer(input, size_in, size_out,kernelSize=5,stride=1, name="conv",dropOut = False):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([kernelSize, kernelSize, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# to specify any arguments with default values do {"argName": argValue,"argName2",argValue2,.......}
def fc_layer(input, size_in, size_out, name="fc",dropOut = False,keepProb = 0.5):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    if dropOut: 
        drop = tf.nn.dropout(tf.matmul(input, w) + b,keepProb)
    else: 
        drop = tf.matmul(input, w) + b
    relu = tf.nn.relu(drop)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", relu)
    return relu



def mnist_model(learning_rate,batchSize, hparam):
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  with sess.as_default():
      # Setup placeholders, and reshape the data
      x = tf.placeholder(tf.float32, shape=[None, pow(inputDim,2)], name="x")
      x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
      tf.summary.image('input', x_image, 3)
      y = tf.placeholder(tf.float32, shape=[None, numOutputs], name="labels")
    
      embedding_input = x
      embedding_size = pow(inputDim,2)
      #define network structure here
      conv_1 = conv_layer(x_image,1,numConvOutputs1,**{"name":"conv1"})
      conv_2 = conv_layer(conv_1,numConvOutputs1,numConvOutputs2,**{"name":"conv2"})
      flatten = tf.reshape(conv_2,[-1,10*10*numConvOutputs2])
      fc_1 = fc_layer(flatten, 10*10*numConvOutputs2,numFcOutputs1,"fc1")
      logits = fc_layer(fc_1, numFcOutputs1, numOutputs, "Output")
    
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
    
      embedding = tf.Variable(tf.zeros([3800, embedding_size]), name="test_embedding")
      assignment = embedding.assign(embedding_input)
      saver = tf.train.Saver()
      
      """Initialise variables, key step, can only make tensorflow objects after this"""
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter(os.path.join(LOGDIR, hparam))
      writer.add_graph(sess.graph)
    
      batchSize = batchSize
      iterations = 3001
      displayNum = 5
      testNum = 3005
#
          
          
      print("Creating dataset tensors...")
      tensorCreation = t.time()
      #create dataset for training and validation
      tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
      tr_data = tr_data.repeat()
      #take a batch of 128
      tr_data = tr_data.batch(batchSize)
      val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))

      val_data = val_data.shuffle(buffer_size=10000)
      #repeat the test dataset infinitely, so that we can loop over its test
      val_data = val_data.repeat()
      #loop over the test value
      val_data = val_data.batch(testLength)
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
      
      #print(tf.one_hot(next_label,numOutputs).eval())
      #print(len(tf.one_hot(next_label,numOutputs).eval()))
      
      for i in range(iterations): #range 2001
          if i % displayNum == 0:
              print('calculating training accuracy... i={}'.format(i))
              tempIm = next_image.eval()
              tempLabel = tf.one_hot(next_label,numOutputs).eval()
              #print(tempIm)
              #print(tempLabel)
              element = 0
              for i in range(len(tempLabel)):
                  if tempLabel[i][element] == 1:
                      im.arrayToImage(tempIm[i],40,40).show()#save(savePath+"{}.png".format(i))
                      
              [train_accuracy, s] = sess.run([accuracy, summ], \
                  feed_dict={x: tempIm, \
                             y: tempLabel})
              
              writer.add_summary(s, i)
          if i % testNum == 0 and i!=0:
              print('did 500, saving')
              sess.run(assignment, \
                       feed_dict={x: next_val_image.eval()[:100],  \
                                  y: tf.one_hot(next_val_label,numOutputs).eval()[:100]})
              saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
          sess.run(train_step, \
                   feed_dict={x: next_image.eval(), \
                              y: tf.one_hot(next_label,numOutputs ).eval()})

def make_hparam_string(learning_rate,batchSize):
  fc_param = "fc=2"
  conv_param = "conv=2"
  return "lr_%.0E,batch_%s,%s,%s" % (learning_rate,batchSize, fc_param, conv_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-7,1E-6,1E-5,1E-4,1E-3]:
      for batchSize in [100]:

        # Include "False" as a value to try different model architectures
            # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate,batchSize)
        print('Starting run for %s' % hparam)
    
    	    # Actually run with the new settings
        mnist_model(learning_rate,batchSize, hparam)


if __name__ == '__main__':
  main()