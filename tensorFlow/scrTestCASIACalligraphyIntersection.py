#%% Imports and paths
import os
import scipy as sp
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#os.chdir('C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling')
os.chdir(os.path.join(gitHubRep,"dataHandling"))
from classFileFunctions import fileFunc as fF 
from classImageFunctions import imageFunc as iF
#os.chdir('C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\tensorFlow')
os.chdir(os.path.join(gitHubRep,"tensorFlow"))
from InputTFRecord import inputs
#set other variables
inputDim = 48
numOutputs= 245#number of outputs in original network

##set paths and file names Seb
#dataPath, LOGDIR, rawDatapath = fF.whichUser("Seb")
#dataPath = 'C:/Users/Sebastian/Desktop/MLChinese'
#relTrainDataPath = 'CASIA/1.0'
#relSavePath = 'Visualising_filters'
#relModelPath = '2conv_100Train_TransferOriginal/Outputs100_LR0.001_Batch128'
#relTransferModelPath = 'transfer_learning/finalLayerCNN_was100Out_run_1/Outputs3866_LR0.001_Batch128'

#set paths and file names Elliot
dataPath, LOGDIR, rawDatapath = fF.whichUser("Elliot")
relTrainDataPath = 'Machine learning data/TFrecord'
relSavePath = 'savedVisualisation'
relModelPath = 'TF_record_CNN/Outputs100_LR0.001_Batch128'
relTransferModelPath = 'transfer_learning/finalLayerCNN_was100Out_run_1/Outputs3866_LR0.001_Batch128'


originalModelName = 'LR0.001_Iter20520_TestAcc0.845.ckpt'
transferModelName = 'LR0.001_Iter19710_TestAcc0.6796875.ckpt'

#import rest of the modules modules
import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image, ImageDraw, ImageFont

tf.reset_default_graph()
numImages = 245 #batch size

#test_tfrecord_filename = 'C:\\Users\\Sebastian\\Desktop\\GitHub\\ML-for-Chinese-Calligraphy\\dataHandling\\calligraphy_CASIA.tfrecords'
test_tfrecord_filename = os.path.join(gitHubRep,"dataHandling")+"/calligraphy_CASIA.tfrecords"

#%% Load in first graph, get the bottlenecks from it
print("FIRST GRAPH")
print("Importing graph.....")
sess = tf.InteractiveSession()
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(originalModelName+".meta")
print("Restoring the session...")
saver.restore(sess,'./'+originalModelName)
print("Getting default graph...")
graph = tf.get_default_graph()
print("Printing graph operations...")
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/dropout/keep_prob:0")
getAccuracy=graph.get_tensor_by_name("accuracy/accuracy:0")
getBottleneck = graph.get_tensor_by_name("dropout/dropout/mul:0")

print("Setting up data....")
start = t.time()
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data":False}
train_image_batch, train_label_batch = inputs('test',test_tfrecord_filename,1000,1,**train_kwargs)
print("took ",t.time()-start," seconds\n")

images,labels=sess.run([train_image_batch,train_label_batch])
bottlenecks = sess.run(getBottleneck, feed_dict={x: images, keep_prob: 1.0})
print("Max label:{}, min label:{}".format(max(labels),min(labels)))

sess.close()
tf.reset_default_graph()


#%% Load in the second graph, feed in the bottlenecks
print("SECOND GRAPH")
print("Now deploying the full network by first loading the last layer from the transfer learn")
print("Importing graph.....")
sess = tf.InteractiveSession()
loadLOGDIR = os.path.join(LOGDIR,relTransferModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(transferModelName+".meta")

print("Restoring the session...")
saver.restore(sess,'./'+transferModelName)
print("Getting default graph...")
graph = tf.get_default_graph()
print("took ",t.time()-start," seconds\n")
print("Assign operations and placeholders......")    
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
getAccuracy=graph.get_tensor_by_name("accuracy/accuracy:0")

accuracyList = []
for i in range(30):
    accuracy = sess.run(getAccuracy, feed_dict={x: bottlenecks,y_:tf.one_hot(labels,3866).eval()})
    accuracyList.append(accuracy)
    

labels = [i-171 for i in labels]
# labels are consistent throughout
# labels and images match

print(len(bottlenecks),len(labels))
accuracy = sess.run(getAccuracy, feed_dict={x: bottlenecks,y_:tf.one_hot(labels,3866).eval()})
print("Accuracy: {}".format(accuracy))
  
sess.close()
tf.reset_default_graph()