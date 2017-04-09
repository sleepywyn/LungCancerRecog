import threading
import tensorflow as tf
import numpy as np
import pandas as pd
import time

class FIFO_Queue:
    BATCH_SIZE = 2
    def __init__(self, capacity, feature_input_shape, label_input_shape, input_data_folder, input_label_file, input_df, sess, coord):
        self.feature_input_shape = feature_input_shape
        self.label_input_shape = label_input_shape
        self.capacity = capacity
        self.input_data_folder = input_data_folder
        self.input_label_file = input_label_file
        self.input_df = input_df
        self.sess = sess
        self.coord = coord
        self.feature_placeholder = tf.placeholder(tf.float32, shape=feature_input_shape)
        self.label_placeholder = tf.placeholder(tf.int32, shape=label_input_shape)

        # create queue and operation
        self.q = tf.FIFOQueue(capacity=capacity, dtypes=[tf.float32, tf.int32], shapes=[feature_input_shape, label_input_shape])
        self.enqueue_op = self.q.enqueue([self.feature_placeholder, self.label_placeholder])
        self.data_sample, self.label_sample = self.q.dequeue()
        self.data_many_sample, self.label_many_sample = self.q.dequeue_many(self.BATCH_SIZE)



    """
    directly read patient from csv file and load image from input folder
    :param self.input_data_folder
    :param self.input_label_file
    """

    def load_and_enqueue_from_file(self):
        i = 1
        with open(self.input_label_file) as label_file:
            next(label_file)
            while True:
                try:
                    line = label_file.next()
                except:
                    break
                if line == '':
                    break  # coord.request_stop()
                patient_id, label_value = line.split(",")
                try:
                    d3_data = np.load(self.input_data_folder + "/" + patient_id + ".npy")
                except:
                    print "Fail to load patient: " + patient_id + " Skipping it."
                    continue
                self.sess.run(self.enqueue_op, feed_dict={self.feature_placeholder: d3_data, self.label_placeholder: label_value})
                i += 1
                print("thread ended for loop " + str(i))

    def dequeue(self):
        return self.q.dequeue()

    """
    enqueue according to a df describing input. This method is used for enqueuing df_train
    :param self.input_df, col: id, cancer
    """
    def enqueue_from_df(self):
	print("Entering enqueue_from_df...")
        dfList = [self.input_df, self.input_df]
        double = pd.concat(dfList)
        for index, row in double.iterrows():
            # print "looping " + str(index)
            if self.coord.should_stop():
                print "queue stop signal received"
                break
            patient_id = row['id']
            label_value = row['label']
            # print(patient_id, label_value)
            # print(self.input_data_folder + "/cubic_" + str(patient_id) + ".npz")
            try:
                image_data = np.load(self.input_data_folder + "/cubic_" + str(patient_id) + ".npz")['arr_0']
            except:
                print("Error loading image...")
                continue
            # print("Ready to Enqueue...")
            if image_data.shape == (36, 36, 36):
            	self.sess.run(self.enqueue_op, feed_dict={self.feature_placeholder: image_data, self.label_placeholder: label_value})
            else:
                print("WARN: Skip 1 data with shape " + str(image_data.shape))
            # print("Enqueued data")

    """
    dequeue one element
    """
    def dequeue_one(self):
    #    print "dequeue started. "
        one_data, one_label = self.sess.run([self.data_sample, self.label_sample])
        return one_data, one_label

    """
    cancel queue op
    """
    def cancel_pending(self):
        op = self.q.close(cancel_pending_enqueues=True)
        self.sess.run(op)

    def dequeue_many(self):
        data, label = self.sess.run([self.data_many_sample, self.label_many_sample])
        # print("Dequeue done...")
        # print(data.shape)
        return data, label

    def close(self):
        self.q.close()
