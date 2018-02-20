# -*- coding: utf-8 -*-
"""
Build and train a neural network using TFRecords as input
Look at the bottom of the file for all the inputs
"""

import tensorflow as tf
import time
import math #to use radians in rotating the image
import random
from classDataManip import makeDir
import os

def decodeTrain(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={'train/image': tf.FixedLenFeature([], tf.string),
                  'train/label': tf.FixedLenFeature([], tf.int64)})
    # tfrecords is saved in raw bytes, need to convert this into usable format
    # May want to save this as tf.float32???
    image = tf.decode_raw(features['train/image'], tf.uint8)
    # Reshape image data into the original shape (try different forms)
#    image1 = tf.reshape(image, [inputDim, inputDim, 1]); #2D no batch
#    image2 = tf.reshape(image, [inputDim**2,1]);         #1D no batch
    """Try with no '1' on the end of array (which denotes RGB or greyscale)"""
    image1 = tf.reshape(image, [inputDim, inputDim]); #2D no batch
    image2 = tf.reshape(image, [inputDim**2]);         #1D no batch
    print(image1)
    print(image2)
    # Cast label data
    label = tf.cast(features['train/label'], tf.int32)
    return image2, label

def decodeTest(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={'test/image': tf.FixedLenFeature([], tf.string),
                  'test/label': tf.FixedLenFeature([], tf.int64)})
    # tfrecords is saved in raw bytes, need to convert this into usable format
    # May want to save this as tf.float32???
    image = tf.decode_raw(features['test/image'], tf.uint8)
    # Reshape image data into the original shape (try different forms)
#    image1 = tf.reshape(image, [inputDim, inputDim, 1]); #2D no batch
#    image2 = tf.reshape(image, [inputDim**2,1]);         #1D no batch
    """Try with no '1' on the end of array (which denotes RGB or greyscale)"""
    image1 = tf.reshape(image, [inputDim, inputDim]); #2D no batch
    image2 = tf.reshape(image, [inputDim**2]);         #1D no batch
    print(image1)
    print(image2)
    # Cast label data
    label = tf.cast(features['test/label'], tf.int32)
    return image2, label

def augment(image, label):
    """Apply distortions to the image, here rotation and translation
    Not included yet"""
    #reshape the image so it has 2D shape
    image = tf.reshape(image, [inputDim, inputDim,1])
    
    #adjust brightness
    image = tf.image.random_brightness(image,0.1)
    
    #rotate image
    degree_angle = random.randint(-10,10) # random integer from -10 to 10
    print("Rotation by {} degrees".format(degree_angle))
    radian = degree_angle * math.pi / 180 #convert to radians
    image = tf.contrib.image.rotate(image,radian)
    
    #translate image
    translate_x = int(np.random.normal(0)) #random integer normally distributed
    translate_y = int(np.random.normal(0)) #random integer normally distributed
    if abs(translate_x) > 2 or abs(translate_y) > 2:
        translate_x = 0 #so we don't get a shift of 5 
        translate_y = 0
    print("Translation by {},{} in x,y".format(translate_x,translate_y))
    image = tf.contrib.image.translate(image,(translate_x,translate_y))
    
    #scale the image
    #scaling doesn't seem to work with how small the lines are
#    resize_scale = inputDim + 0
#    image = tf.image.resize_images(image,tf.constant([inputDim+0,inputDim+5]))
#    image = tf.image.resize_image_with_crop_or_pad(image,inputDim,inputDim)
   
    #reshape the image back into 1D
    image = tf.reshape(image, [inputDim**2])
    
    return image, label

def normalize(image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label

# Creates a dataset that reads all of the examples from two files.
def inputs(trainType,tfrecord_filename,batch_size,num_epochs,normalize=False,augment_images=False):
    """If num_epochs is set to 0, repeat infinitely"""
    #filenames = [tfrecord_filename]
    filenames = tfrecord_filename
    dataset = tf.data.TFRecordDataset(filenames)
    if num_epochs == 0: #if set to 0, repeat infinitely
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)
    print(dataset)
    
    if trainType == 'train':
        dataset = dataset.map(decodeTrain)  # Parse the record into tensors.
        if augment_images == True:   
            print("Augmenting the images")
            dataset = dataset.map(augment)
    elif trainType == 'test':
        #do not augment testing data! Only need to augment training data
        dataset = dataset.map(decodeTest)
    else:
        raise ValueError("trainType not specified properly as train or test")
    
    if normalize == True:
        dataset=dataset.map(normalize) #normalize the image values to be between -0.5 and 0.5
        
    dataset = dataset.shuffle(1000 + 3 * batch_size) #shuffle the order of the images
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.
# Initialize `iterator` with training data.


def run_training():
    tf.reset_default_graph()
        
    train_image_batch, train_label_batch = inputs('train',train_tfrecord_filename,train_batch_size,num_epochs)
    test_image_batch, test_label_batch = inputs('test',test_tfrecord_filename,test_batch_size,0)

    #build a graph here
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
    
    #Define the placeholders for the images and labels
    # 'None' used to be batch_size << haven't tested None yet
    x = tf.placeholder(tf.float32, [None, inputDim**2], name="images")
    x_image = tf.reshape(x, [-1, inputDim, inputDim, 1]) #to show example images
    tf.summary.image('input', x_image, 4) # Show 4 examples of output images
    y_ = tf.placeholder(tf.float32, [None,num_output], name="labels")
    
    with tf.name_scope('fc1'):
        """Fully connected layer, maps features to the number of outputs"""
        w_fc = weight_variable([inputDim**2,num_output])
        b_fc = bias_variable([num_output])
        # calculate the convolution
        y_conv = tf.matmul(x, w_fc) + b_fc
        tf.summary.histogram("activations", y_conv)
        
    with tf.name_scope("xent"):    
        cross_entropy = tf.reduce_mean(\
                            tf.nn.softmax_cross_entropy_with_logits(\
                                labels=y_, \
                                logits=y_conv))
        tf.summary.scalar("xent",cross_entropy)
        
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # what fraction of bools was correct? Cast to floating point...
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)

    #merge all summaries for tensorboard
    mergedSummaryOp = tf.summary.merge_all()
    # Create a saver to save these summary operations
    saver = tf.train.Saver()
    
    #this is the operation that initialises all the graph variables
#    init_op = tf.group(tf.global_variables_initializer(),\
#                       tf.local_variables_initializer())
    
    
        
    sess = tf.Session()
    #with tf.Session() as sess:
        
    tf.global_variables_initializer().run()    
        
    # Create writers
    train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/test')
    test_writer.add_graph(sess.graph)
    
    #initialise the variables
    #sess.run(init_op)
    
    start_time = time.time()  
    try:
        step = 0
        while True: #train until we run out of epochs
              
            if step % 100 == 0:
                train_accuracy, train_summary = sess.run([accuracy, mergedSummaryOp], \
                             feed_dict={x: train_image_batch.eval(), \
                                        y_: tf.one_hot(train_label_batch,num_output).eval()})
                train_writer.add_summary(train_summary, step)
                print('Step: {}, Training accuracy = {:.3}'.format(step, train_accuracy))
            
            if step % 500 == 0:
                print("Testing the net...")
                test_accuracy, test_summary = sess.run([accuracy,mergedSummaryOp], \
                               feed_dict={x: test_image_batch.eval(),\
                                          y_: tf.one_hot(test_label_batch,num_output).eval()})
                test_writer.add_summary(test_summary, step)
                saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TestAcc{:.3}.ckpt".\
                                              format(learning_rate,step,test_accuracy)))
                print('Step: {}, Test accuracy = {:.3}'.format(step, test_accuracy))
            
            #run the training
            sess.run(train_step, feed_dict={x: train_image_batch.eval(),\
                                            y_: tf.one_hot(train_label_batch,num_output).eval()})              
            step += 1
    except tf.errors.OutOfRangeError:
        duration = time.time() - start_time
        print('Done {} epochs, {} steps, took {:.3} mins.'.\
              format(num_epochs,step,duration/60))  
    
    train_writer.close()
    test_writer.close()
            
def main():
    run_training()

inputDim = 48
num_output_list = [10]
num_epoch_list = [10]
train_batch_size_list = [128]
learning_rate_list = [1E-3]
test_batch_size = 500

savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved runs\\'
localPath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0'

name_of_run = 'test'

for num_output in num_output_list:
    train_tfrecord_filename = \
        localPath+'\\train'+str(num_output)+'.tfrecords'
    test_tfrecord_filename = \
        localPath+'\\test'+str(num_output)+'.tfrecords'
    for num_epochs in num_epoch_list:
        for train_batch_size in train_batch_size_list:
            for learning_rate in learning_rate_list:
                LOGDIR = makeDir(savePath,name_of_run,num_output,\
                                     learning_rate,train_batch_size)
                main()
    
    