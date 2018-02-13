import time
import numpy as np
import tensorflow as tf

# Constants used for dealing with the files, matches convert_to_records.
TFRecord_path = 'C:\\Users\\Sebastian\\Desktop\\MLChinese\\CASIA\\1.0\\train.tfrecords'
inputDim = 48
num_epochs = 1
batch_size = 128
learningRate = 1e-3
numOutputs = 10

#%% Image decoding and manipulation, and reading the inputs

def decode(serialized_example,inputDim):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={'train/image': tf.FixedLenFeature([], tf.string),
                  'train/label': tf.FixedLenFeature([], tf.int64)})
    # tfrecords is saved in raw bytes, need to convert this into usable format
    # May want to save this as tf.float32???
    image = tf.decode_raw(features['train/image'], tf.uint8)
    # Reshape image data into the original shape (try different forms)
    image1 = tf.reshape(image, [inputDim, inputDim, 1]);    print(image1) #2D no batch
    image2 = tf.reshape(image, [inputDim**2,1]);            print(image2) #1D no batch
    image3 = tf.reshape(image, [batch_size,inputDim**2,1]); print(image3) #1D batch
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

def read_file(filename_queue,inputDim):
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)
    image, label = decode(record_string,inputDim)
    image, label = augment(image, label)
    image, label = normalize(image, label)
    return image, label

def input_pipeline(TFRecord_path,inputDim,batch_size,num_epochs):
    """Reads input data num_epochs times.
      This function creates a one_shot_iterator, meaning that it will only iterate
      over the dataset once. On the other hand there is no special initialization
      required.
    """
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(
            [TFRecord_path], num_epochs=num_epochs, shuffle=True)
    image, label = read_file(filename_queue,inputDim)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        allow_smaller_final_batch=True)
    return image_batch, label_batch

image_batch, label_batch = input_pipeline(TFRecord_path,inputDim,batch_size,num_epochs)
print(image_batch)
print(label_batch)

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
print("Generating input pipeline...")
image_batch, label_batch = input_pipeline(TFRecord_path,inputDim,batch_size,num_epochs)
print(label_batch)
# Initialise all variables
print("Initialising variables...")
tf.global_variables_initializer().run()
print("Evaluating labels...")
print(sess.run(label_batch))

#print(sess.run(image_batch))


#%%


data = np.arange(1, 100 + 1)
data_input = tf.constant(data)

batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=10, capacity=100, min_after_dequeue=10, allow_smaller_final_batch=True)
batch_no_shuffle = tf.train.batch([data_input], enqueue_many=True, batch_size=10, capacity=100, allow_smaller_final_batch=True)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(i, sess.run([batch_shuffle, batch_no_shuffle]))
    coord.request_stop()
    coord.join(threads)



#%% The training program
#
#def run_training():
#    """Train for a number of steps."""
#    
#    # Tell TensorFlow that the model will be built into the default Graph.
#    with tf.Graph().as_default():
#        # Input images and labels.
#        image_batch, label_batch = input_pipeline(
#                TFRecord_path, inputDim, batch_size=128,num_epochs=1)
#
#    def weight_variable(shape):
#        """weight_variable generates a weight variable of a given shape."""
#        initial = tf.truncated_normal(shape, stddev=0.1)
#        tf.summary.histogram("weights", initial)
#        return tf.Variable(initial)
#    
#    def bias_variable(shape):
#        """bias_variable generates a bias variable of a given shape."""
#        initial = tf.constant(0.1, shape=shape)
#        tf.summary.histogram("biases", initial)
#        return tf.Variable(initial)
#
#    # Define the placeholders for images and labels
#    x = tf.placeholder(tf.float32, [None, inputDim**2], name="images")
#    x_image = tf.reshape(x, [-1, inputDim, inputDim, 1])
#    # Show 4 examples of output images
#    tf.summary.image('input', x_image, 4)
#    y_ = tf.placeholder(tf.float32, [None,numOutputs], name="labels")
#    
#    with tf.name_scope('fc2'):
#        """Fully connected layer 2, maps 1024 features to the number of outputs"""
#        #1024 inputs, 10 outputs
#        W_fc2 = weight_variable([inputDim**2, numOutputs])
#        b_fc2 = bias_variable([numOutputs])
#        # calculate the convolution
#        y_conv = tf.matmul(x, W_fc2) + b_fc2
#        tf.summary.histogram("activations", y_conv)
#    
#    
#    # Calculate entropy on the raw outputs of y then average across the batch size
#    with tf.name_scope("xent"):
#        cross_entropy = tf.reduce_mean(\
#                            tf.nn.softmax_cross_entropy_with_logits(\
#                                labels=y_, \
#                                logits=y_conv))
#        tf.summary.scalar("xent",cross_entropy)
#        
#    with tf.name_scope("train"):
#        train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
#    
#    with tf.name_scope("accuracy"):
#        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#        # what fraction of bools was correct? Cast to floating point...
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        tf.summary.scalar("accuracy",accuracy)
#    
#    # Merge all summary operators
#    mergedSummaryOp = tf.summary.merge_all()
#    # Create a saver to save these summary operations AND the embedding
#    saver = tf.train.Saver()
#
#    # The op for initializing the variables.
#    init_op = tf.group(tf.global_variables_initializer(),
#                       tf.local_variables_initializer())
#
#    # Create a session for running operations in the Graph.
#    with tf.Session() as sess:
#        # Initialize the variables (the trained variables and the
#        # epoch counter).
#        sess.run(init_op)
#        try:
#            step = 0
#            while True:  #train until OutOfRangeError
#                start_time = time.time()
#
#                # Run one step of the model.  The return values are
#                # the activations from the `train_op` (which is
#                # discarded) and the `loss` op.  To inspect the values
#                # of your ops or variables, you may include them in
#                # the list passed to sess.run() and the value tensors
#                # will be returned in the tuple from the call.
#                _, loss_value = sess.run([train_step, loss])
#
#                duration = time.time() - start_time
#
#                # Print an overview fairly often.
#                if step % 100 == 0:
#                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
#                                                       duration))
#                    step += 1
#        except tf.errors.OutOfRangeError:
#            print('Done training for %d epochs, %d steps.' % (num_epochs,
#                                                          step))
#
#
#def main(_):
#    run_training()
#
