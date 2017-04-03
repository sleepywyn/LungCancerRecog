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

def read_luna_csv(label_file):
    df = pd.read_csv(label_file)
    grouped = df.groupby("seriesuid").agg({
        'coordX' : lambda x: list(x),
        'coordY': lambda x: list(x),
        'coordZ': lambda x: list(x),
        'diameter_mm': lambda x: list(x)
    })
    # result = grouped.apply(lambda row: [row['coordX'][0]] + [row['coordY'][0]], axis=1)
    grouped['nodule'] = grouped.apply(lambda row: concatRow(row), axis=1)
    return grouped[['nodule']]

def concatRow(row):
    l = []
    for x in range(len(row['coordX'])):
        l.append([row['coordX'][x]] + [row['coordY'][x]] + [row['coordZ'][x]] + [row['diameter_mm'][x]])
    return l

def load_npz(file):
    return np.load(file)['arr_0']

def split(df, ratio):
    samp_num = int(df.shape[0] * ratio)
    indices = df.index
    rows_num = random.sample(range(df.shape[0]), samp_num)
    rows_ix = map(lambda x: indices[x], rows_num)

    df_train = df.ix[rows_ix]
    df_test = df.drop(rows_ix)
    return df_train, df_test


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

def read_prediction(file_path):
    df = pd.read_csv(file_path)
    return df

def luna_unet_gen(df, data_folder):
    # df = read_luna_csv(csv_path)
    for index, row in df.iterrows():
        #row['nodule']
        print("Start fetching patient: " + index)
        image_path = data_folder + "/" + index + "_lung_img.npz"
        ground_truth_path = data_folder + "/" + index + "_nodule_mask.npz"
        try:
            input_3d, target_3d = load_npz(image_path), load_npz(ground_truth_path)
        except: continue

        gen_list = []
        slice_num = target_3d.shape[0]
        for i in range(slice_num):
            if np.count_nonzero(target_3d[i]) > 0:
                gen_list.append(i)
        gen_list += list(np.random.randint(low=20, high=slice_num - 20, size=len(gen_list)))
        # for i in range(input_3d.shape[0]):
        for i in gen_list:
            print("Returning data for z index: " + str(i))
            yield (np.expand_dims(np.expand_dims(input_3d[i], axis=0), axis=0), np.expand_dims(np.expand_dims(target_3d[i], axis=0), axis=0))
        # yield (np.expand_dims(np.expand_dims(np.mean(input_3d, axis=0), axis=0), axis=0), np.expand_dims(np.expand_dims(np.mean(target_3d, axis=0), axis=0), axis=0))

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





