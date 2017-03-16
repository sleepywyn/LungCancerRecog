# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import fifo
import threading
import simple_reader as sr
"""
    Tensor board visualization method
    """
summaries_dir = "/Users/ibm/GitRepo/tensor_board"

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

"""
    Computation function
    """
def one_hot(x):
    return np.array([1, 0]) if x == 0 else np.array([0, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')  # [batch_size, mov_z, mov_x, mov_y, input channel]

def max_pool_2X2X2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()
#### simple_load test images ###
df_train, df_test = sr.read_and_split("./data/stage1_labels.csv", 0.9)
test_images_list, test_labels_list = sr.read_image_from_split(df_test, "./data/d3_images_mid")
test_images, test_labels = np.asarray(test_images_list), np.stack(map(one_hot, test_labels_list))


# image_list, label_list = simple_reader.read_file("./data/test.csv", "./data/d3_slices")
coord = tf.train.Coordinator()
queue = fifo.FIFO_Queue(capacity=60, feature_input_shape=[32, 64, 64],
                        label_input_shape=[],
                        input_data_folder="./data/d3_images_mid",
                        input_label_file="",
                        input_df=df_train,
                        sess=sess,
                        coord=coord)

t = threading.Thread(target=queue.enqueue_from_df, name="enqueue")
t.start()

x  = tf.placeholder(tf.float32, [None, 32, 64, 64], name='x')
y_ = tf.placeholder(tf.float32, [None, 2],  name='y_')

x_image = tf.reshape(x, [-1, 32, 64, 64, 1]) #[-1, width, height, color channel]  -1 means that the length in that dimension is inferred

##------- Layer1 -------##
with tf.name_scope('weights_conv1'):
    W_conv1 = weight_variable([5, 5, 5, 1, 16]) # [filter_size, filter_size, num_input_channels, num_filters (k value, sometimes called output channel. This is NOT RGB channel)]
    variable_summaries(W_conv1)
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2X2X2(h_conv1)

##------- Layer2 -------##
with tf.name_scope('weights_conv2'):
    W_conv2 = weight_variable([5, 5, 5, 16, 64])
    variable_summaries(W_conv2)
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2X2X2(h_conv2)

##------- Fully Connected Layer1 -------##
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
W_fc1 = weight_variable([8 * 16 * 16 * 64, 200])
b_fc1 = bias_variable([200])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 16 * 16 * 64])  # flatten the layer. prepare for fully connected layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##------- Drop Out Layer -------##
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##------- Final Layer -------##
W_fc2 = weight_variable([200, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##------- Train -------##
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # get the max value index on dimension one
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # accuracy function
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(summaries_dir + '/test', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(45):
    # batch_xs, batch_ys = mnist.train.next_batch(50)
    # deq_xs, deq_ys = queue.dequeue_one()
    deq_xs, deq_ys = queue.dequeue_many()
    # batch_xs, batch_ys = image_list[i].reshape((1, 160, 160)), label_list[i]
    # batch_ys = np.array([[1, 0]]) if batch_ys == 0 else np.array([[0, 1]])
    
    batch_xs, batch_ys = deq_xs.reshape(10, 32, 64, 64), map(one_hot, deq_ys.reshape(10, 1))
    # batch_xs, batch_ys = deq_xs.reshape(1, 32, 64, 64), deq_ys
    # batch_ys = np.array([[1, 0]]) if batch_ys == 0 else np.array([[0, 1]])
    
    if i % 50 == 0 and i != 0:            # alternative. calculating accuracy regarding to test set.
        summary, acc = sess.run([merged, accuracy], feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        print(i, acc)
    else:
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(feed_dict={
                                       x: test_images, y_: test_labels, keep_prob: 1.0}))

coord.request_stop()
print coord.should_stop()
queue.cancel_pending()
coord.join([t])
