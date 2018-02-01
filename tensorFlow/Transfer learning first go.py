# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:38:17 2018

@author: ellio
"""


#%% Imports, set directories
import os
gitHubRep = os.path.normpath(os.getcwd() + os.sep + os.pardir)

import tensorflow as tf
os.chdir(os.path.join(gitHubRep,"dataHandling/"))
from classFileFunctions import fileFunc as fF 
#os.chdir(os.path.join(gitHubRep,"tensorFlow/"))
#from classDataManip import subSet,oneHot,makeDir,Data,createSpriteLabels
dataPath, LOGDIR = fF.whichUser("Elliot")
#%%
 
##create a graph with 2 variables and an operation

##Prepare to feed input, i.e. feed_dict and placeholders
#w1 = tf.placeholder("float", name="w1")
#w2 = tf.placeholder("float", name="w2")
#b1= tf.Variable(2.0,name="bias")
#feed_dict ={w1:4,w2:8}
# 
##Define a test operation that we will restore
#w3 = tf.add(w1,w2)
#w4 = tf.multiply(w3,b1,name="op_to_restore")
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
# 
##Create a saver object which will save all the variables
#saver = tf.train.Saver()
# 
##Run the operation by feeding input
#print (sess.run(w4,feed_dict))
##Prints 24 which is sum of (w1+w2)*b1 
# 
##Now, save the graph
#os.chdir(LOGDIR)
#saver.save(sess, './my_test_model',global_step=1000)


# load the graph and run with different values
os.chdir(LOGDIR)
sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
 
 
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
 
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}
 
#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0") 
print (sess.run(op_to_restore,feed_dict))
#This will print 60 which is calculated 
#using new values of w1 and w2 and saved value of b1

# create new op to add on to saced graph
newOp=tf.multiply(op_to_restore,2)
print(sess.run(newOp,feed_dict))