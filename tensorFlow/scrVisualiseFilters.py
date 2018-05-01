# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 12:11:18 2018
A script to output a visulaisation of a models filters, will also eventually use it to deploy a network
and produce an embedding etc.  
@author: ellio
"""


#%% Imports and paths
import os
import scipy as sp
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)# find github path
#gitHubRep = 'C:/Users/Sebastian/Desktop/GitHub/ML-for-Chinese-Calligraphy'
#import own functions and classes
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
from classImageFunctions import imageFunc as iF
os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
from InputTFRecord import inputs
#set other variables
inputDim = 48
numOutputs= 3866#number of outputs in original network

#set paths and file names
dataPath, LOGDIR, rawDatapath = fF.whichUser("Seb")
dataPath = 'C:/Users/Sebastian/Desktop/MLChinese'
#relTrainDataPath = "Machine learning data/TFrecord" #path of training data relative to datapath in classFileFunc
relTrainDataPath = 'CASIA/1.0'
#relSavePath = "savedVisualisation" #path for saved images relative to dataPath
relSavePath = 'Visualising_filters'
#relModelPath = 'TF_record_CNN/Outputs100_LR0.001_Batch128'# path of loaded model relative to LOGDIR
relModelPath = '2conv_100Train_TransferOriginal/Outputs100_LR0.001_Batch128'
relTransferModelPath ='transfer_learning/finalLayerCNN_was100Out_run_1/Outputs3866_LR0.001_Batch128'
#originalModelName="LR0.001_Iter20520_TestAcc0.845.ckpt"#name of ckpt file with saved original model
originalModelName = 'LR0.001_Iter20520_TestAcc0.845.ckpt'
transferModelName ='LR0.001_Iter19710_TestAcc0.6796875.ckpt' #name of model that contains the retrained last layer
saveName = "Transfer_learning_test"#name for saved images

#import rest of the modules modules
import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
import itertools
import PIL.ImageOps
from PIL import Image, ImageDraw, ImageFont

tf.reset_default_graph()
numImages = 475 #batch size?

# Go to the directory where you have the numpy file containing the list of characters
os.chdir(gitHubRep)
all_chars = np.load("List_of_chars_NUMPY.npy")
chinese_only = all_chars[171:]

#%%Func defs
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

def sideBySide(charPrediction, reshapedImage):
    img = Image.new('RGB', (48, 48), (255,255,255))
    draw = ImageDraw.Draw(img) 
    simsum_font = ImageFont.truetype('simsun.ttc',48) #font must be supported
    draw.text((1,1), charPredictions[i][0], font = simsum_font, fill = "#000000")
              
    imgTotal = Image.new('RGB', (96,48), (255,255,255))
    imgTotal.paste(img, (0,0))
    charImage = Image.fromarray( np.asarray( imagesReshape[i], dtype="uint8"), "L" )
    charImage_invert = PIL.ImageOps.invert(charImage)
    imgTotal.paste(charImage_invert, (48,0))
    print("Saving {}".format(i))
    imgTotal.save(os.path.join(dataPath,relSavePath)+"\\Images_predictions_calligraphy\\{}.png".format(charPredictions[i][1]))        
        
#%% import data
#train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'train'+str(numOutputs)+'.tfrecords')
#test_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'test'+str(numOutputs)+'.tfrecords')
#train_tfrecord_filename = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\calligraphy.tfrecords'
train_tfrecord_filename = os.path.join(os.path.join(dataPath,relTrainDataPath),'calligraphy_greyscaled.tfrecords')
#%% Open a tensorflow session
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
print("Initialising all variables...")
# Initialise all variables
loadLOGDIR = os.path.join(LOGDIR,relModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(originalModelName+".meta")

print("Restoring the session...")
saver.restore(sess,'./'+originalModelName)

print("Getting default graph...")
graph = tf.get_default_graph()
print("took ",t.time()-start," seconds\n")
#print(graph.get_operations())

print("Set up data....")
start = t.time()
train_kwargs = {"normalize_images": True, "augment_images": False, "shuffle_data":False}
train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,numImages,1,**train_kwargs)
print("took ",t.time()-start," seconds\n")

print("Assign operations and placeholders......")  
start = t.time()  
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
keep_prob=graph.get_tensor_by_name("dropout/dropout/keep_prob:0")
conv1Activations = graph.get_tensor_by_name("conv_1/Relu:0")
conv1Weights = graph.get_tensor_by_name("conv_1/Variable:0")
conv2Activations = graph.get_tensor_by_name("conv_2/Relu:0")
conv2Weights = graph.get_tensor_by_name("conv_2/Variable:0")

#conv3Activations = graph.get_tensor_by_name("conv_3/Relu:0")
#conv3Weights = graph.get_tensor_by_name("conv_3/Variable:0")
#conv4Activations = graph.get_tensor_by_name("conv_4/Relu:0")
#conv4Weights = graph.get_tensor_by_name("conv_4/Variable:0")
#
#conv5Activations = graph.get_tensor_by_name("conv_5/Relu:0")
#conv5Weights = graph.get_tensor_by_name("conv_5/Variable:0")
#conv6Activations = graph.get_tensor_by_name("conv_6/Relu:0")
#conv6Weights = graph.get_tensor_by_name("conv_6/Variable:0")

#accuracy=graph.get_tensor_by_name("accuracy/Mean:0")
getBottleneck = graph.get_tensor_by_name("dropout/dropout/mul:0")
output = graph.get_tensor_by_name("add:0")
print("took ",t.time()-start," seconds\n")

#%% extract  feature maps
images,labels=sess.run([train_image_batch,train_label_batch])
print(labels)
print("Max label:{}, min label:{}".format(max(labels),min(labels)))
layer1Activations=sess.run(conv1Activations,feed_dict={x: images, keep_prob: 1.0})
layer1Weights=sess.run(conv1Weights)
layer2Activations=sess.run(conv2Activations,feed_dict={x: images, keep_prob: 1.0})
layer2Weights=sess.run(conv2Weights)

#layer3Activations=sess.run(conv3Activations,feed_dict={x: images, keep_prob: 1.0})
#layer3Weights=sess.run(conv3Weights)
#layer4Activations=sess.run(conv4Activations,feed_dict={x: images, keep_prob: 1.0})
#layer4Weights=sess.run(conv4Weights)
#
#layer5Activations=sess.run(conv5Activations,feed_dict={x: images, keep_prob: 1.0})
#layer5Weights=sess.run(conv5Weights)
#layer6Activations=sess.run(conv6Activations,feed_dict={x: images, keep_prob: 1.0})
#layer6Weights=sess.run(conv6Weights)

bottlenecks = sess.run(getBottleneck,feed_dict={x: images, keep_prob: 1.0})

#%%process feature maps
layer1Activations=removeAndSwapAxes(layer1Activations)
layer2Activations=removeAndSwapAxes(layer2Activations)

#layer3Activations = removeAndSwapAxes(layer3Activations)
#layer4Activations = removeAndSwapAxes(layer4Activations)
#
#layer5Activations = removeAndSwapAxes(layer5Activations)
#layer6Activations = removeAndSwapAxes(layer6Activations)

#save activations, weights, bottlenecks and outputs
fF.saveNPZ(os.path.join(dataPath,relSavePath),"features_raw_{}".format(numImages)+saveName+".npz",\
           images=np.reshape(images,(numImages,inputDim,inputDim)),\
           layer1=layer1Activations,weight1=layer1Weights,\
           layer2=layer2Activations,weight2=layer2Weights,\
           
#           layer3 = layer3Activations, weight3 = layer3Weights, \
#           layer4 = layer4Activations, weight4 = layer4Weights, \
           
           bottlenecks=bottlenecks)

activations_list = [layer1Activations, layer2Activations] #\
                    #, layer3Activations, layer4Activations \
                    #, layer5Activations, layer6Activations]
weights_list = [layer1Weights, layer2Weights]# \
                   # , layer3Weights, layer4Weights \
                   # , layer5Weights, layer6Weights]

#%%find for each feature the image that creaates the highest activation.
def maximum_activation(layerActivations):
    layerHighest = []
    activationTot = np.sum(layerActivations,axis=(2,3))
    maxIndices = np.argmax(activationTot,axis=0)
    for i in range(0,layerActivations.shape[1]):
            layerHighest.append(layerActivations[int(maxIndices[i])][i])
    return layerHighest

for i in activations_list:
    print("Layer {} activations...".format(activations_list.index(i)))
    layerHighest = maximum_activation(i)
    show_activations(layerHighest, len(layerHighest[0]), len(layerHighest), os.path.join(dataPath,relSavePath),\
                 "layer{}Features.jpg".format(activations_list.index(i)), show=True, save=True)

def convert_weight_filters(layerWeights):
    weight_filters = [layerWeights[:,:,:,i] for i in range(layerWeights.shape[3])]
    weight_filters = [np.sum(i,axis=2) for i in weight_filters]
    weight_upscaled = [sp.misc.imresize(weight_filters[i],10.0) for i \
                        in range(layerWeights.shape[3])]
    return weight_upscaled

for i in weights_list:
    print("Layer {} weights...".format(weights_list.index(i)))
    weights_upscaled = convert_weight_filters(i)
    show_activations(weights_upscaled, len(weights_upscaled[0]), len(weights_upscaled),\
                     os.path.join(dataPath,relSavePath), 'layer{}Weights.jpg'.format(weights_list.index(i)),\
                     show=True, save=True)
    
sess.close()
tf.reset_default_graph()
#%%    
#deploy network
print("Now deploying the full network by first loading the last layer from the transfer learn")
print("Importing graph.....")
start = t.time()
sess = tf.InteractiveSession()
print("Initialising all variables...")
# Initialise all variables
loadLOGDIR = os.path.join(LOGDIR,relTransferModelPath)
os.chdir(loadLOGDIR)
saver = tf.train.import_meta_graph(transferModelName+".meta")

print("Restoring the session...")
saver.restore(sess,'./'+transferModelName)

print("Getting default graph...")
graph = tf.get_default_graph()
print("took ",t.time()-start," seconds\n")
#print(graph.get_operations())
print("Assign operations and placeholders......")  
start = t.time()  
x=graph.get_tensor_by_name("images:0")
y_=graph.get_tensor_by_name("labels:0")
getAccuracy=graph.get_tensor_by_name("accuracy/accuracy:0")
getPredictions = graph.get_tensor_by_name("accuracy/ArgMax:0")
print("took ",t.time()-start," seconds\n")
#get accuracy and predictions for batch
#accuracy = sess.run(getAccuracy,feed_dict={x: bottlenecks,y_:tf.one_hot(labels,})
predictions = sess.run(getPredictions,feed_dict={x:bottlenecks})
sess.close()
tf.reset_default_graph()
charPredictions = []
for i in range (0,len(predictions)):
    charPredictions.append((chinese_only[predictions[i]],labels[i]))
    
imagesReshape=np.reshape(images,(numImages,inputDim,inputDim))
fF.saveNPZ(os.path.join(dataPath,relSavePath),"Images_+_predictions_calligraphy"+saveName+".npz",\
           images=imagesReshape,\
           predictions=charPredictions)
for i in range(0,numImages):
#    plt.imshow(imagesReshape[i],cmap = 'gray')
#    print(charPredictions[i])
#    input("Wait for iamge to load and press enter")
    sideBySide(charPredictions[i], imagesReshape[i])
    
