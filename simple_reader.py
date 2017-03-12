import numpy as np
import pandas as pd
import random

def decode_label(label):
    return int(label)

"""
read data into memory given a label file
"""
def read_file(label_file, image_folder):
    data_file = []
    labels = []
    with open (label_file, "r") as f:
        next(f)
        for line in f:
            patient_id, label = line.split(",")
            try:
                path = image_folder + "/" + patient_id + ".npy"
                d3_data = np.load(path)
            except:
                continue
            data_file.append(d3_data)
            labels.append(decode_label(label))
        return data_file, labels


"""
split the label file according to specified ratio
"""
def read_and_split(label_file, ratio):
    df = pd.read_csv(label_file)
    samp_num = int(df.shape[0] * ratio)
    a = range(df.shape[0])
    rows_ix = random.sample(range(df.shape[0]), samp_num)
    df_train = df.ix[rows_ix]
    df_test = df.drop(rows_ix)
    return df_train, df_test

"""
read data into memory given a list
This method is used specifically for loading test image data(only once) for cross validation
"""
def read_image_from_split(split_df, image_folder):
    print split_df
    images = []
    labels = []
    for index, row in split_df.iterrows():
        path = image_folder + "/" + row['id'] + ".npy"
        try:
            image_data = np.load(path)
        except:
            continue
        label = row['cancer']
        images.append(image_data)
        labels.append(label)
    return images, labels

def read_and_sample(file_path):
    df = pd.read_csv(file_path)


"""
generate random split index
param: seed, number, replace, ratio
"""
# def gen_index(rdm, n_train, ratio, replace=False):
# 	n_samp = int(n_train * ratio)
# 	if replace:
# 		index_base = rdm.randint(n_train, size=n_samp)
# 		index_meta = [i for i in range(n_train) if i not in index_base]
# 	else:
# 		rand_num = rdm.uniform(size=n_train)
# 		index_base = [i for i in range(n_train) if rand_num[i] <= ratio]
# 		index_meta = [i for i in range(n_train) if rand_num[i] > ratio]
# 	return index_base, index_meta


"""
test
"""
# df_train, df_test = read_and_split("./data/stage1_labels.csv", 0.9)
# test_images, test_labels = read_image_from_split(df_test, "./data/d3_slices_large")
# print df_test





