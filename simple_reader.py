import numpy as np
import pandas as pd
import random
from scipy.signal import argrelextrema

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

def read_npz_image_from_split(split_df, image_folder):
    print split_df
    images = []
    labels = []
    for index, row in split_df.iterrows():
        path = image_folder + "/cubic_" + str(row['id']) + ".npz"
        try:
            image_data = np.load(path)['arr_0']
        except:
            continue
        label = int(row['label'])
        if image_data.shape == (36, 36, 36):
            images.append(image_data)
            labels.append(label)
    return images, labels

def read_prediction(file_path):
    df = pd.read_csv(file_path)
    return df

def luna_unet_gen(df, data_folder):
    # df = read_luna_csv(csv_path)
    pool_counter = 0
    image_pool = []

    for index, row in df.iterrows():
        print("Start fetching patient: " + index)
        image_path = data_folder + "/" + index + "_lung_img.npz"
        ground_truth_path = data_folder + "/" + index + "_nodule_mask.npz"
        try:
            input_3d, target_3d = load_npz(image_path), load_npz(ground_truth_path)
        except: continue

        gen_list = []
        slice_num = target_3d.shape[0]
        # for i in range(slice_num):
        #     num_zero = np.count_nonzero(target_3d[i])
        #     if np.count_nonzero(target_3d[i]) > 50:
        #         gen_list.append((i, num_zero))
        # gen_list = gen_list[0:1]
        # random.shuffle(gen_list)
        # for i in range(input_3d.shape[0]):
        # for index, _ in gen_list:
        #     print("Returning data for z index: " + str(index))
        #     yield (np.expand_dims(np.expand_dims(input_3d[index], axis=0), axis=0), np.expand_dims(np.expand_dims(target_3d[index], axis=0), axis=0))


        for i in range(slice_num):
            num_zero = np.count_nonzero(target_3d[i])
            gen_list.append(num_zero)

        # gen_list += list(np.random.randint(low=20, high=slice_num - 20, size=int(len(gen_list) / 2)))
        # gen_list.sort(key=lambda x: x[1], reverse=True)
        gen_array = np.array(gen_list)
        flagged = np.r_[True, gen_array[1:] >= gen_array[:-1]] & np.r_[gen_array[:-1] >= gen_array[1:], True]

        mask_0 = gen_array != 0
        flagged = np.logical_and(flagged, mask_0)

        for i in range(len(flagged)):
            if flagged[i] == True:
                print("Returning data for z index: " + str(i))
                image_pool.append((input_3d[i], target_3d[i]))
                pool_counter += 1
        if(pool_counter > 100):
            random.shuffle(image_pool)
            for (input, target) in image_pool:
                yield (np.expand_dims(np.expand_dims(input, axis=0), axis=0), np.expand_dims(np.expand_dims(target, axis=0), axis=0))
            pool_counter = 0
            image_pool = []

        # yield (np.expand_dims(np.expand_dims(input_3d[gen_list[len(gen_list) / 2]], axis=0), axis=0), np.expand_dims(np.expand_dims(target_3d[gen_list[len(gen_list) / 2]], axis=0), axis=0))
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


'''
This function is used to convert the world coordinates to voxel coordinates using
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

"""
test
"""
# df_train, df_test = read_and_split("./data/stage1_labels.csv", 0.9)
# test_images, test_labels = read_image_from_split(df_test, "./data/d3_slices_large")
# print df_test
