import tensorflow as tf
import numpy as np
import fifo
import simple_reader as sr
import threading

"""
Computation function
"""

def one_hot(x):
    return np.array([1, 0]) if x == 0 else np.array([0, 1])


"""
Layer definition
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

graph_feat_num = 2048
graph_num = 21
step_size = 1e-5
mid_num = 256

q_capacity = 4
dequeue_size = 2
train_seg = 0.9
iter_num = 2000 / dequeue_size

summaries_dir = "/opt/LungCancerRecog/board"
result_path = "/opt/LungCancerRecog/data/pred.csv"


def main():
    sess = tf.Session()
    tf.InteractiveSession()

    df_train, df_test = sr.read_and_split("./data/stage1_labels.csv", train_seg)
    test_features_list, test_labels_list = sr.read_image_from_split(df_test, "./out")
    test_features, test_labels = np.asarray([a.flatten() for a in test_features_list]), np.stack(map(one_hot, test_labels_list))


    coord = tf.train.Coordinator()
    queue = fifo.FIFO_Queue(capacity=q_capacity, feature_input_shape=[graph_num, graph_feat_num],
                            label_input_shape=[],
                            input_data_folder="./out",
                            input_label_file="",
                            input_df=df_train,
                            sess=sess,
                            coord=coord)

    t = threading.Thread(target=queue.enqueue_from_df, name="enqueue")
    t.start()


    x = tf.placeholder(tf.float32, [None, graph_feat_num * graph_num], name='x')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_')

#    W_fc_mid = weight_variable([graph_feat_num * graph_num, graph_num * mid_num])
#    bias_mid = weight_variable([graph_num * mid_num])

#    midMap = tf.nn.relu(tf.matmul(x, W_fc_mid) + bias_mid)

    W_fc = weight_variable([graph_num * graph_feat_num, 2])
    bias = weight_variable([2])

    y_predict = tf.matmul(x, W_fc) + bias

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_predict))
        tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(step_size).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                          tf.argmax(y_, 1))  # get the max value index on dimension one
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # accuracy function
    tf.summary.scalar('accuracy', accuracy)

    ##------- Predict Op-------##
    with tf.name_scope("final_prediction"):
        prediction_op = tf.nn.softmax(logits=y_predict)
    ##-------End of Predict Op-------##

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test', sess.graph)
    sess.run(tf.global_variables_initializer())


    for i in range(iter_num):
        print("Loop starts")
        deq_xs, deq_ys = queue.dequeue_many()
        batch_xs, batch_ys = deq_xs.reshape(dequeue_size, graph_num * graph_feat_num), map(one_hot, deq_ys.reshape(dequeue_size, 1))

        if i % 4 == 0 and i != 0:  # alternative. calculating accuracy regarding to test set.
            summary, acc = sess.run([merged, accuracy], feed_dict={x: test_features, y_: test_labels})
            test_writer.add_summary(summary, i)
            print(i, acc)
        else:
            print(i)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
            train_writer.add_summary(summary, i)

    coord.request_stop()
    print coord.should_stop()
    queue.cancel_pending()
    coord.join([t])
    queue.close()


    print "===========Prediction start===================================="

    df_prediction = sr.read_prediction("./data/stage1_sample_submission.csv")
    
    coord_predict = tf.train.Coordinator()
    predict_queue = fifo.FIFO_Queue(capacity=q_capacity,
                                     feature_input_shape=[graph_num, graph_feat_num],
                                     label_input_shape=[],
                                     input_data_folder="./out",
                                     input_label_file="",
                                     input_df=df_prediction,
                                     sess=sess,
                                     coord=coord_predict)
    t2 = threading.Thread(target=predict_queue.enqueue_from_df, name="enqueue_prediction_data")
    t2.start()
    
    for i in range(df_prediction.shape[0] / dequeue_size):
        deq_xs, deq_ys = predict_queue.dequeue_many()
        batch_xs, batch_ys = deq_xs.reshape(dequeue_size, graph_num * graph_feat_num), map(one_hot, deq_ys.reshape(dequeue_size, 1))
    
        batch_prediction = sess.run(prediction_op, feed_dict={x: batch_xs, y_: batch_ys})
        for j in range(batch_prediction.shape[0]):
            df_prediction.ix[i * dequeue_size + j, 1] = batch_prediction[j, 1]
    
    print df_prediction
    df_prediction.to_csv(result_path)
    coord_predict.request_stop()
    predict_queue.cancel_pending()
    coord_predict.join([t2])
    predict_queue.close()


main()
