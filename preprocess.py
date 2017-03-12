import os
import dicom
import numpy as np
import pandas as pd
import scipy.ndimage
# http://opencvpython.blogspot.com/2012/05/install-opencv-in-windows-for-python.html
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 1. Load Dicom files
def load_slices(path):
    try:
        slices = [dicom.read_file(path + "/" + s) for s in os.listdir(path)]
    except:
        print("Damaged dicom file: " + path)
        return
    # NOTE: slices[0].ImagePositionPatient returns ['-145.500000', '-158.199997', '-316.200012']; Use the 3rd place value (z-axis) to sort
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slices_thickness = np.abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    except:
        print("WARN: NO ImagePositionPatient[2] property, use SliceLocation for %s" % slices[0])
        slices_thickness = np.abs(slices[1].SliceLocation - slices[0].SliceLocation)
    # NOTE: create new metadata, the SliceThickness, and store it in dicom dataset
    for s in slices:
        s.SliceThickness = slices_thickness
    return slices

# 2. Convert pixel values to HU
def convert_hu(slices):
    d3_image = np.stack([s.pixel_array for s in slices])
    d3_image = d3_image.astype(np.int16)
    print("INFO: 3d image has shape: " + str(d3_image.shape))
    print("INFO: %i pixels in %i pixel are out of bound and has value -2000. Set them to 0 " % (
        np.sum(d3_image == -2000), d3_image.shape[0] * d3_image.shape[1] * d3_image.shape[2]))
    # NOTE: set -2000 pixels to 0
    d3_image[d3_image == -2000] = 0
    # NOTE: convert pixel values to HU by multiplying slope and adding intercept
    for s_num in range(len(slices)):
        slope = slices[s_num].RescaleSlope
        intercept = slices[s_num].RescaleIntercept
        if slope != 1:
            d3_image[s_num] = slope * d3_image[s_num].astype(np.float64)
            d3_image[s_num] = d3_image[s_num].astype(np.int16)
        d3_image[s_num] += np.int16(intercept)
    print("INFO: 3d image of HU has been generated")
    return d3_image

# 3. Resampling
def resample(image, slices, new_spacing=[1, 1, 1]):
    # NOTE: get the size of the cube for a set of slices of a person. x * y size is stored in PixelSpacing, z size is stored in SliceThickness.
    spacing = np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)
    print("INFO: 3d image has spacing " + str(spacing))  # [2.5  0.59765601  0.59765601]
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    # NOTE: http://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode="nearest")

    return image, new_spacing

# 4. Lung Segment
# 4.1. Region Growing
# 4.2. Connected Component Analysis

# 5. Normalization
def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    return image

# 6. zero centering
def zero_center(image):
    PIXEL_MEAN = 0.25
    image -= PIXEL_MEAN
    return image

# 7. resize
def resize(image, axis):
    IMG_PX_SIZE = 64
    Z_PX_SIZE = 32
    if axis == 0:
        image_resized_list = []
        for s_num in range(image.shape[0]):
            slice_resized = cv2.resize(np.array(image[s_num]), (IMG_PX_SIZE, IMG_PX_SIZE))
            image_resized_list.append(slice_resized)
        image_resized = np.stack(image_resized_list)
        print("INFO: Resized 3d image on x,y-axis to " + str(image_resized.shape))
    elif axis == 1:
        image_resized_list = []
        for s_num in range(image.shape[1]):
            slice_resized = cv2.resize(np.array(image[:, s_num, :]), (IMG_PX_SIZE, Z_PX_SIZE))
            image_resized_list.append(slice_resized)
        image_resized = np.stack(image_resized_list).transpose(1, 0, 2)
        print("INFO: Resized 3d image on z-axis to " + str(image_resized.shape))
    else:
        print("ERROR: Parameter axis in resize function is not correct")
    return image_resized

# 8. show 3d image
def plot_3d(image, threshold=-300):
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

############################
##          Util          ##
############################
def preprocess(input_folder="./data/sample_images", output_folder="./d3_images"):
    patients = os.listdir(input_folder)
    count = 0
    for patient in patients:
        patient_folder = input_folder + "/" + patient
        patient_slices = load_slices(patient_folder)
        if patient_slices is None:
            continue
        patient_d3_image = convert_hu(patient_slices)
        patient_d3_image_resample, new_spacing = resample(patient_d3_image, patient_slices)
        print("INFO: Before resampling, patient's image shape is " + str(patient_d3_image.shape))
        print("INFO: After  resampling, patient's image shape is " + str(patient_d3_image_resample.shape))
        print("INFO: New spacing is " + str(new_spacing))  # [ 1.  0.9999996  0.9999996]
        patient_d3_image_resample_clean = zero_center(normalize(patient_d3_image_resample))
        patient_d3_image_resample_clean_resized = resize(patient_d3_image_resample_clean, 0)
        patient_d3_image_resample_clean_resized_xyz = resize(patient_d3_image_resample_clean_resized, 1)
        # plot_3d(patient_d3_image_resample_clean_resized, 400)  # preview 3d image
        # plt.show()
        # plot_3d(patient_d3_image_resample_clean_resized_new, 400)  # preview 3d image
        # plt.show()

        output_patient = output_folder + "/" + patient
        count += 1
        np.save(output_patient, patient_d3_image_resample_clean_resized_xyz)
        print "saving: No." + str(count) + " patient. id: " + patient
        print "=============================================================="

def decode_label(label):
    return int(label)

def read_label_file(file):
    data_file_paths = []
    labels = []
    with open(file, "r") as f:
        next(f)
        for line in f:
            filepath, label = line.split(",")
            data_file_paths.append(filepath)
            labels.append(decode_label(label))
        return data_file_paths, labels

def observe_thickness(path):
    slices = [dicom.read_file(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    for i in range(len(slices) - 1):
        slices_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i + 1].ImagePositionPatient[2])
        print slices_thickness

############################
##          Main          ##
############################
if __name__ == '__main__':
    input_folder = "./data/images/stage1"
    ############## single example ############
    # patients = os.listdir(input_folder)
    # first_patient = patients[1]
    # print("INFO: Fetch the 1st patient %s" % first_patient)
    #
    # first_patient_folder = input_folder + "/" + first_patient
    # first_patient_slices = load_slices(first_patient_folder)
    # first_patient_d3_image = convert_hu(first_patient_slices)
    #
    # first_patient_d3_image_resample, new_spacing = resample(first_patient_d3_image, first_patient_slices)
    # print("INFO: Before resampling, 1st patient's image shape is " + str(first_patient_d3_image.shape))
    # print("INFO: After  resampling, 1st patient's image shape is " + str(first_patient_d3_image_resample.shape))
    # print("INFO: New spacing is " + str(new_spacing))  # [ 1.  0.9999996  0.9999996]
    # first_patient_d3_image_resample_clean = zero_center(normalize(first_patient_d3_image_resample))
    # first_patient_d3_image_resample_clean_resized = resize(first_patient_d3_image_resample_clean)
    # plot_3d(first_patient_d3_image_resample, 0.5) # preview 3d image

    ############## preprocess ############
    # observe_thickness("./data/images/stage1/b8bb02d229361a623a4dc57aa0e5c485")
    # observe_thickness("./data/images/stage1/a4dc34f2731b62f60c6c390a39fe11b2")

    preprocess(input_folder, "./data/d3_images_mid")

    ############## load data ############
    # plot_3d(first_patient_d3_image_resample_clean, -3000)  # preview 3d image
    # plt.hist(first_patient_d3_image.flatten(), bins=80, color='c')
    # plt.show()
    # plt.imshow(first_patient_d3_image[80], cmap=plt.cm.gray)
    # plt.show()
    # plt.imshow(first_patient_d3_image_resample[80], cmap=plt.cm.gray)
    # plt.show()
    # plt.imshow(first_patient_d3_image_resample_clean[80], cmap=plt.cm.gray)
    # plt.show()