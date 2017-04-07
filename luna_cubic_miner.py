import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
import preprocess_luna as pre_luna


input_folder = "./data_luna/subset0"
train_csv_path = "./data_luna/CSVFILES/annotations.csv"
output_folder = "./data_luna/out_cubic"

'''
This function takes the numpy array of CT_Scan and a list of coords from
which voxels are to be cut. The list of voxel arrays are returned. We keep the
voxels cubic because per pixel distance is same in all directions.
'''

def get_patch_from_list(lung_img, coords, window_size=10):
    shape = lung_img.shape
    output = []
    lung_img = lung_img + 1024
    for i in range(len(coords)):
        patch = lung_img[coords[i][0] - 18: coords[i][0] + 18,
                coords[i][1] - 18: coords[i][1] + 18,
                coords[i][2] - 18: coords[i][2] + 18]
        output.append(patch)

    return output

'''
Sample a random point from the image and return the coordinates.
'''

def get_point(shape):
    x = np.random.randint(50, shape[2] - 50)
    y = np.random.randint(50, shape[1] - 50)
    z = np.random.randint(20, shape[0] - 20)
    return np.asarray([z, y, x])

'''
This function reads the training csv file which contains the CT Scan names and
location of each nodule. It cuts many voxels around a nodule and takes random points as
negative samples. The voxels are dumped using pickle. It is to be noted that the voxels are
cut from original Scans and not the masked CT Scans generated while creating candidate
regions.
'''

def create_data(path, train_csv_path):
    trainY = []
    count = 0
    with open(train_csv_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            print("Processing loop: " + str(count) + "\n")
            row = line.split(',')

            if os.path.isfile(path + "/" + row[0] + '.mhd') == False:
                continue

            # lung_img = sitk.GetArrayFromImage(sitk.ReadImage(path + "/" +  row[0] + '.mhd'))
            lung_img, origin, spacing = pre_luna.load_itk(path + "/" +  row[0] + '.mhd')

            coords = []
            for i in range(-3, 4, 3): #3
                for j in range(-3, 4, 3): #3
                    for k in range(-2, 3, 2): #3
                        coords.append([float(row[3]) + k, float(row[2]) + j, float(row[1]) + i])
                        trainY.append(True)
            coords = world_2_voxel(coords, origin, spacing).tolist()
            for i in range(8):
                coords.append(get_point(lung_img.shape))
                trainY.append(False)


            trainX = get_patch_from_list(lung_img, coords)
            print (trainX[0].shape, len(trainX))

            # save voxels here
            for cubic in trainX:
                np.savez_compressed(output_folder + "/cubic_" + str(count), np.asarray(cubic))
                count += 1

        data = {'id': range(len(trainY)), 'label': trainY}
        df = pd.DataFrame(data = data)
        df.to_csv(output_folder + "/cubic_labels.csv", index=False)


            # pickle.dump(np.asarray(trainX), open('traindata_' + str(counter) + '_Xtrain.p', 'wb'))
            # pickle.dump(np.asarray(trainY, dtype=bool), open('traindata_' + str(counter) + '_Ytrain.p', 'wb'))

'''
This function is used to convert the world coordinates to voxel coordinates using
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = (stretched_voxel_coordinates / spacing).astype(int)
    return voxel_coordinates


if __name__ == '__main__':
    create_data(input_folder, train_csv_path)
