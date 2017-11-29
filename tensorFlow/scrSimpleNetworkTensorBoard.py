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

#user = "Elliot"
user = "Seb"

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

whichTest = 4

LOGDIR = LOGDIR + str(datetime.date.today()) + '/test-{}'.format(whichTest)
#make a directory
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

os.chdir(funcPath)
from classFileFunctions import fileFunc as fF 
os.chdir("..")

#%%Get the data
#set ration of data to be training and testing
trainRatio = 0.70

print("splitting data...")
startTime=t.time()
#file to open

dataPath = dataPathSeb

fileName="CharToNumList_10"
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

#%% Check images are correct
#from PIL import Image
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

numOutputs = 3755
inputDim = 40

def mnist_model(learning_rate, hparam):
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
      
      """Have separate writers for training and testing"""
      train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'train'))
      train_writer.add_graph(sess.graph)
      
      test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'test'))
      test_writer.add_graph(sess.graph)
      
      """Work on the embedding"""
#      config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
#      embedding_config = config.embeddings.add()
#      embedding_config.tensor_name = embedding.name
#      embedding_config.sprite.image_path = SPRITES
#      embedding_config.metadata_path = LABELS
#      # Specify the width and height of a single thumbnail.
#      embedding_config.sprite.single_image_dim.extend([28, 28])
#      tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
      
      print("Creating dataset tensors...")
      tensorCreation = t.time()
      #create dataset for training and validation
      tr_data = tf.data.Dataset.from_tensor_slices((trainImages,trainLabels))
      tr_data = tr_data.repeat()
      #take a batch of 128
      tr_data = tr_data.batch(128)
      val_data = tf.data.Dataset.from_tensor_slices((testImages,testLabels))
      val_data = val_data.batch(len(testLabels))
      val_data = val_data.shuffle(buffer_size=10000)
      #repeat the test dataset infinitely, so that we can loop over its test
      val_data = val_data.repeat()
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
      
      for i in range(3001):
#          if i % 30 == 0:  # Record summaries and test-set accuracy
#              summary, acc = sess.run([merged_summary_op, accuracy], \
#                                      feed_dict={x: next_val_image.eval(),  \
#                                                 y: tf.one_hot(next_val_label,3755).eval()})
#              test_writer.add_summary(summary, i)
#              print('Accuracy at step %s: %s' % (i, acc))
#          else:  # Record train set summaries, and train
#            if i % 100 == 99:  # Record execution stats
#              summary, _ = sess.run([merged_summary_op, train_step],
#                                    feed_dict={x: next_image.eval(),  \
#                                               y: tf.one_hot(next_label,3755).eval()})
#              train_writer.add_summary(summary, i)
#            else:  # Record a summary
#              summary, _ = sess.run([merged_summary_op, train_step], \
#                                    feed_dict={x: next_image.eval(),  \
#                                               y: tf.one_hot(next_label,3755).eval()})
#              train_writer.add_summary(summary, i)
          
          if i % 30 == 0:
              print('calculating training accuracy... i={}'.format(i))
              train_accuracy, train_summary = sess.run([accuracy, merged_summary_op], \
                  feed_dict={x: next_image.eval(), \
                             y: tf.one_hot(next_label,3755).eval()})
              train_writer.add_summary(train_summary, i)
          if i % 100 == 0:
              print('calculating test accuracy and saving')
              sess.run(assignment, \
              #assign, test_accuracy, test_summary = sess.run([assignment, accuracy, merged_summary_op], \
                       feed_dict={x: next_val_image.eval(),  \
                                  y: tf.one_hot(next_val_label,3755).eval()})
              #test_writer.add_summary(test_summary,i)
              saver.save(sess, os.path.join(LOGDIR, "model.ckpt{}".format(learning_rate)), i)
          sess.run(train_step, \
                   feed_dict={x: next_image.eval(), \
                              y: tf.one_hot(next_label,3755).eval()})
      train_writer.close()
      test_writer.close()
    
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

	    # Actually run with the new settings
    mnist_model(learning_rate, hparam)


if __name__ == '__main__':
  main()