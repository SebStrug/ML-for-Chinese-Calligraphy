# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:06:57 2018

@author: ellio
"""
import tensorflow as tf   
import os     
# a class to hold inportat info on a layer        
class layer:
    def __init__(self,layerType,outputShape,output):
        self.layerType=layerType
        self.outputs=outputShape
        self.output = output
    def getOutputShape(self):
        return self.outputShape
    def getType(self):
        return self.layerType
    def getOutput(self):
        return self.output

class NeuralNet:
    def __init__(self):
        self.layers = []
        self.numFC=0
        self.numConv=0
        self.numReshape=0
        self.numDropout=0
        self.numRelu=0
        self.numPool=0
        
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
    
    def addInputLayer(self,inputDim):
        tf.reset_default_graph()
        with tf.name_scope("Input"):
            h_input = tf.placeholder(tf.float32, [None, inputDim**2], name="images")
        self.layers.append(layer("input",[1,1,inputDim**2],h_input))
        
    def addConvLayer(self,outputs,kernelSize=5,stride=1):
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        prevType = self.layers[len(self.layers)-1].getType()
        allowedPrevTypes=["reshape", "conv", "relu", "pool", "input"]
        if prevType not in allowedPrevTypes:
            print('Error: Cannot place layer of type [conv] after layer of type ',prevType)
        else:
            self.numConv+=1
            with tf.name_scope('Conv{}'.format(self.numConv)):
                with tf.name_scope("Weights"):
                    w_conv = self.weight_variable([kernelSize,kernelSize, prevOutputShape[2], outputs])
                with tf.name_scope("Bias"):
                    b_conv = self.bias_variable([outputs])
                with tf.name_scope("Convolve"):
                    h_conv = tf.nn.conv2d(prevOutput, w_conv, strides=[1, stride, stride, 1], padding='SAME') + b_conv
                    tf.summary.histogram("activations", h_conv)      
            self.layers.append(layer("conv",[prevOutputShape[0],prevOutputShape[1],outputs],h_conv))
     
    def addPoolLayer(self,kernelSize=2,stride = 2):
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        prevType = self.layers[len(self.layers)-1].getType()
        allowedPrevTypes=["conv", "input", "dropout", "relu"]
        if prevType not in allowedPrevTypes:
            print("Error: Cannot place layer of type [pool] after layer of type ",prevType)
        else:
            self.numpool+=1
            with tf.name_scope('Pool{}'.format(self.numPool)):
                h_pool=tf.nn.max_pool(prevOutput, ksize=[1, kernelSize, kernelSize, 1],strides=[1, stride, stride, 1], padding='SAME')
            self.layers.append(layer("pool",[prevOutputShape[0]/2,prevOutputShape[1]/2,prevOutputShape[2]],h_pool))    
        
    def addFCLayer(self,outputs):
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        prevType = self.layers[len(self.layers)-1].getType()
        allowedPrevTypes=["reshape", "fc", "relu", "input"]
        if prevType not in allowedPrevTypes:
            print("Error: Cannot place layer of type [fc] after layer of type ",prevType)
        else:
            self.numFC+=1
            with tf.name_scope('FC{}'.format(self.numFC)):
                with tf.name_scope("Weights"):
                    w_fc = self.weight_variable([prevOutputShape[2], outputs])
                with tf.name_scope("Bias"):
                    b_fc = self.bias_variable([outputs])
                with tf.name_scope("MatMul"):
                    h_fc = tf.matMul(prevOutput, w_fc) + b_fc
                    tf.summary.histogram("activations", h_fc)      
            self.layers.append(layer("conv",[1,1,outputs],h_fc))
    
    def addReluLayer(self):
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        prevType = self.layers[len(self.layers)-1].getType()
        allowedPrevTypes=["reshape", "fc", "conv", "input", "dropout"]
        if prevType not in allowedPrevTypes:
            print("Error: Cannot place layer of type [relu] after layer of type ",prevType)
        else:
            self.numRelu+=1
            with tf.name_scope('Relu{}'.format(self.numRelu)):
                h_relu=tf.nn.relu(prevOutput)
            self.layers.append(layer("relu",prevOutputShape,h_relu))
            
    def addReshapeLayer(self,outputShape,axis=-1):
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        prevType = self.layers[len(self.layers)-1].getType()
        self.numReshape+=1
        with tf.name_scope('Reshape{}'.format(self.numReshape)):
            h_reshape=tf.reshape(prevOutput, [axis,outputShape[0],outputShape[1],outputShape[2]])
        self.layers.append(layer("reshape",outputShape,h_reshape))
    
    def addDropoutLayer(self):
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        prevType = self.layers[len(self.layers)-1].getType()
        allowedPrevTypes=["reshape", "fc", "conv", "input", "relu"]
        if prevType not in allowedPrevTypes:
            print("Error: Cannot place layer of type [dropout] after layer of type ",prevType)
        else:
            self.numDropout+=1
            with tf.name_scope('Dropout{}'.format(self.numDropout)):
                keep_prob = tf.placeholder(tf.float32,name="keep_prob")
                h_dropout=tf.nn.dropout(prevOutput,keep_prob)
            self.layers.append(layer("dropout",prevOutputShape,h_dropout))
    
    def addOutputLayer(self,numOutputs,savePath,saveName):
        y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")
        prevOutput = self.layers[len(self.layers)-1].getOutput()
        prevOutputShape = self.layers[len(self.layers)-1].getOutputShape()
        with tf.name_scope("xent"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prevOutput),name="xent")
            tf.summary.scalar("xent",cross_entropy)
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(prevOutput, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
            tf.summary.scalar("accuracy",accuracy)
        self.layers.append(layer("output",prevOutputShape,prevOutput))
        
        mergedSummaryOp = tf.summary.merge_all(name="merged_summary")
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver.save(sess, os.path.join(savePath, saveName))
            
        