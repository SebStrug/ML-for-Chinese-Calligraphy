# -*- coding: utf-8 -*-
"""Second attempt"""
import tensorflow as tf
import time
from classDataManip import makeDir
import os

tfrecord_filename = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\train.tfrecords'

def decode(serialized_example):
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

def augment(image, label):
    # OPTIONAL: Could apply distortions
    return image, label

def normalize(image, label):
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label

# Creates a dataset that reads all of the examples from two files.
def inputs(tfrecord_filename,batch_size,num_epochs):
    #filenames = [tfrecord_filename]
    filenames = tfrecord_filename
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat(num_epochs)
    print(dataset)
    
    dataset = dataset.map(decode)  # Parse the record into tensors.
    #dataset = dataset.map(augment)
    dataset=dataset.map(normalize)
    
    dataset = dataset.shuffle(1000 + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.
# Initialize `iterator` with training data.


def run_training():
    tf.reset_default_graph()
    with tf.Graph().as_default():
        
        image_batch, label_batch = inputs(tfrecord_filename,batch_size,num_epochs)
    
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
        x = tf.placeholder(tf.float32, [batch_size, inputDim**2], name="images")
        x_image = tf.reshape(x, [-1, inputDim, inputDim, 1]) #to show example images
        tf.summary.image('input', x_image, 4) # Show 4 examples of output images
        y_ = tf.placeholder(tf.float32, [batch_size,numOutputs], name="labels")
        
        with tf.name_scope('fc1'):
            """Fully connected layer, maps features to the number of outputs"""
            w_fc = weight_variable([inputDim**2,numOutputs])
            b_fc = bias_variable([numOutputs])
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
            train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
        
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
        init_op = tf.group(tf.global_variables_initializer(),\
                           tf.local_variables_initializer())
            
            
        with tf.Session() as sess:
    
            #initialise the variables
            sess.run(init_op)
            
            # Create writers
            train_writer = tf.summary.FileWriter(os.path.join(LOGDIR)+'/train')
            train_writer.add_graph(sess.graph)
            start_time = time.time()  
            try:
                step = 0
                while True: #train until we run out of epochs
                      
                    if step % 5 == 0:
                        train_accuracy, train_summary =sess.run([accuracy, mergedSummaryOp], \
                                     feed_dict={x: image_batch.eval(), \
                                                y_: tf.one_hot(label_batch,numOutputs).eval()})
                        train_writer.add_summary(train_summary, step)
                        saver.save(sess, os.path.join(LOGDIR, "LR{}_Iter{}_TrainAcc{}.ckpt".\
                                                      format(learningRate,step,train_accuracy)))
                        
                        print('Step: {}, accuracy = {:.3}'.format(step, train_accuracy))
                    
                    #print(tf.one_hot(label_batch,10).eval())
                    sess.run(train_step, feed_dict={x: image_batch.eval(),\
                                                    y_: tf.one_hot(label_batch,numOutputs).eval()})              
                    
                    step += 1
            except tf.errors.OutOfRangeError:
                duration = time.time() - start_time
                print('Done {} epochs, {} steps, took {:.3} mins.'.\
                      format(num_epochs,step,duration/60))  
            
            train_writer.close()

inputDim = 48
numOutputs = 10
num_epochs = 6
batch_size = 2
learningRate = 1E-3
savePath = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\Saved runs'
whichTest = 1
def main():
    LOGDIR = makeDir(savePath,whichTest,numOutputs,learningRate,batch_size,1)
    run_training()

main()
    
    
    