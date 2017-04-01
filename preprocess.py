import os
import dicom
import numpy as np
import pandas as pd
from scipy import ndimage
# http://opencvpython.blogspot.com/2012/05/install-opencv-in-windows-for-python.html
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters
from skimage.segmentation import clear_border
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool
from functools import partial
# import pretrain_mx

input_folder = "./data/stage1"
output_folder = "./data/out_origin"
output_seg_folder = "./data/out_origin"
thread_num = 5
mx = False

IMG_PX_SIZE = 224
Z_PX_SIZE = 182

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
    image = ndimage.interpolation.zoom(image, real_resize_factor, mode="nearest")

    return image, new_spacing

# 4. Lung Segment
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
	    return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < 604
    #binary = np.array(im < 604, dtype=np.int8)
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = morphology.disk(2)
    binary = morphology.binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = morphology.disk(10)
    binary = morphology.binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = filters.roberts(binary)
    binary = ndimage.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    return im

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
def preprocess(patient):
    try:
        print("INFO: Processing image for patient " + str(patient))
        patient_folder = input_folder + "/" + patient
        patient_slices = load_slices(patient_folder)
        if patient_slices is None:
            print("WARN: No slices for patient " + str(patient))
            return
        patient_d3_image = convert_hu(patient_slices)
        patient_d3_image_resample, new_spacing = resample(patient_d3_image, patient_slices)
        print("INFO: Before resampling, patient's image shape is " + str(patient_d3_image.shape))
        print("INFO: After  resampling, patient's image shape is " + str(patient_d3_image_resample.shape))
        print("INFO: New spacing is " + str(new_spacing))  # [ 1.  0.9999996  0.9999996]
    #    patient_mask_image = segment_lung_mask(first_patient_d3_image_resample)
    #    print("INFO: Generated binary image during segmentation")
        patient_d3_image_resample_clean = zero_center(normalize(patient_d3_image_resample))
        patient_d3_image_resample_clean_resized = resize(patient_d3_image_resample_clean, 0)
        # patient_d3_image_resample_clean_resized_xyz = resize(patient_d3_image_resample_clean_resized, 1)
        # plot_3d(patient_d3_image_resample_clean_resized, 400)  # preview 3d image
        # plt.show()

        output_patient = output_folder + "/" + patient
        np.save(output_patient, patient_d3_image_resample_clean_resized)
        print("INFO: Saving ndarray of patient %s ... ..." % patient)
        print "=============================================================="
    except:
        print("bad sample encountered: " + patient)

def preprocess_segment(patient):

    print("INFO: Processing segment image for patient " + str(patient))
    patient_folder = input_folder + "/" + patient
    patient_slices = load_slices(patient_folder)
    if patient_slices is None:
        print("WARN: No slices for patient " + str(patient))
        return
    for inx, slice in enumerate(patient_slices):
        patient_slices[inx].pixel_array = get_segmented_lungs(slice.pixel_array)
    print("INFO: Applied binary mask during segmentation")
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
    
    output_patient = output_seg_folder + "/" + patient
    np.save(output_patient, patient_d3_image_resample_clean_resized_xyz)
    print("INFO: Saving segmented ndarray of patient %s ... ..." % patient)
    print "=============================================================="

def preprocess_segment_pretrain(pretrained_model, patient):
    try:
        print("INFO: Processing segment image and do pretraining for patient " + str(patient))
        patient_folder = input_folder + "/" + patient
        patient_slices = load_slices(patient_folder)
        if patient_slices is None:
            print("WARN: No slices for patient " + str(patient))
            return
        for inx, slice in enumerate(patient_slices):
            patient_slices[inx].pixel_array = get_segmented_lungs(slice.pixel_array)
        print("INFO: Applied binary mask during segmentation")
        patient_d3_image = convert_hu(patient_slices)
        patient_d3_image_resample, new_spacing = resample(patient_d3_image, patient_slices)
        print("INFO: Before resampling, patient's image shape is " + str(patient_d3_image.shape))
        print("INFO: After  resampling, patient's image shape is " + str(patient_d3_image_resample.shape))
        print("INFO: New spacing is " + str(new_spacing))  # [ 1.  0.9999996  0.9999996]

        #patient_d3_image_resample_clean = zero_center(normalize(patient_d3_image_resample))
        patient_d3_image_resample_clean_resized = resize(patient_d3_image_resample, 0)
        #patient_d3_image_resample_clean_resized_xyz = resize(patient_d3_image_resample_clean_resized, 1)

        #pretrain_mx.calc_features(pretrained_model, patient, patient_d3_image_resample_clean_resized_xyz, output_seg_folder)
        output_patient = output_seg_folder + "/" + patient
        np.save(output_patient, patient_d3_image_resample_clean_resized)
        print("INFO: Saving segmented feature of patient %s ... ..." % patient)
        print "=============================================================="
    except:
        print("WARN: Bad sample encountered: " + patient)
	
def preprocess_segment_pretrain_mx(pretrained_model, patient):
    print("INFO: Processing segment image and do pretraining for patient " + str(patient))
    patient_folder = input_folder + "/" + patient
    patient_slices = load_slices(patient_folder)
    if patient_slices is None:
        print("WARN: No slices for patient " + str(patient))
        return
    for inx, slice in enumerate(patient_slices):
        patient_slices[inx].pixel_array = get_segmented_lungs(slice.pixel_array)
    print("INFO: Applied binary mask during segmentation")
    patient_d3_image = convert_hu(patient_slices)
    patient_d3_image_resample, new_spacing = resample(patient_d3_image, patient_slices)
    print("INFO: Before resampling, patient's image shape is " + str(patient_d3_image.shape))
    print("INFO: After  resampling, patient's image shape is " + str(patient_d3_image_resample.shape))
    print("INFO: New spacing is " + str(new_spacing))  # [ 1.  0.9999996  0.9999996]

    #patient_d3_image_resample_clean = zero_center(normalize(patient_d3_image_resample))
    patient_d3_image_resample_clean_resized = resize(patient_d3_image_resample, 0)
    patient_d3_image_resample_clean_resized_xyz = resize(patient_d3_image_resample_clean_resized, 1)

    pretrain_mx.calc_features(pretrained_model, patient, patient_d3_image_resample_clean_resized_xyz, output_seg_folder)

    print("INFO: Saving segmented feature of patient %s ... ..." % patient)
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
    patients = os.listdir(input_folder)
    # pretrained_model = pretrain_mx.get_extractor()
    pretrained_model = ""
    if mx:
        func = partial(preprocess_segment_pretrain_mx, pretrained_model)
    else:
        func = partial(preprocess_segment_pretrain, pretrained_model)
    pool = Pool(thread_num)
    pool.map(func, patients)
