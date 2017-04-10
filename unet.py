from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from keras.layers import UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import simple_reader as sr
import numpy.ma as ma

# change the loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

'''
The wrong UNET model in tutorial is compiled in this function.
'''
def unet_model_simple():
    inputs = Input((1, 512, 512))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    # conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    # conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    # conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    # conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)

    # UpConv layer means UpSampling then Convolution
    # up6 = concatenate([Conv2D(512, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)), conv4], axis=1)
    up6 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(up6)
    # conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)

    # up7 = concatenate([Conv2D(256, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)), conv3], axis=1)
    up7 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up7)
    # conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)

    # up8 = concatenate([Conv2D(128, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)), conv2], axis=1)
    up8 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up8)
    # conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)

    # up9 = concatenate([Conv2D(64, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)), conv1], axis=1)
    up9 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up9)
    # conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format='channels_first')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


'''
The UNET model is compiled in this function.
'''
def unet_model():
    inputs = Input((1, 512, 512))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    # conv5 = Dropout(0.0)(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)

    # UpConv layer means UpSampling then Convolution
    up6 = concatenate([Conv2D(512, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)), conv4], axis=1)
    # up6 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(up6)
    # conv6 = Dropout(0.0)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)

    up7 = concatenate([Conv2D(256, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)), conv3], axis=1)
    # up7 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(up7)
    conv7 = Dropout(0.1)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)

    up8 = concatenate([Conv2D(128, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)), conv2], axis=1)
    # up8 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)

    up9 = concatenate([Conv2D(64, (2, 2), activation='relu', padding='same', data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)), conv1], axis=1)
    # up9 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format='channels_first')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=5e-6), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def train_model(annotations, data_folder, output, model):
    all_data = sr.read_luna_csv("./data_luna/CSVFILES/annotations.csv")
    df_train, df_test = sr.split(all_data, 1)
    train_generator = sr.luna_unet_gen(df_train, "./data_luna/out")
    test_generator = sr.luna_unet_gen(df_test, "./data_luna/out")
    # model = unet_model()
    model.fit_generator(generator=train_generator, steps_per_epoch=1, epochs=400, validation_data=None)
    model.save(output)

def load_model(path, model):
    model.load_weights(path)
    return model


def predict_nodule_mask(df_prediction, model, data_folder, output_folder):
    for index, row in df_prediction.iterrows():
        if index > 2:
            return
        patientid = row['id']
        print("Begin prediction for patient " + str(index) + " id: " + str(patientid))
        try:
            patient_3d = sr.load_npz(data_folder + "/" + patientid + ".npz")
        except:
            continue
        patient_3d = patient_3d + 1024        #
        nodule_mask_list = []
        for slice in patient_3d:
            expanded = np.expand_dims(np.expand_dims(slice, axis=0), axis=0)
            nodule_mask = model.predict(expanded)[0][0] # predict and shrink dims
            nodule_mask_list.append(nodule_mask)
        nodule_mask_3d = np.asarray(nodule_mask_list)
        masked_3d = ma.masked_where(nodule_mask_3d != 1, patient_3d)
        prediction = masked_3d.filled(0)
        np.savez_compressed(output_folder + "/" + str(patientid), prediction)
        # print prediction[0]
        # print prediction.shape
        # print np.count_nonzero(prediction)
        # expanded = np.expand_dims(patient_3d, axis=1) # (z, kernel=1, 512, 512)
        # print expanded.shape
        # nodule_mask = model.predict(expanded)
        # print nodule_mask.shape
        # return


############################
##          Main          ##
############################
if __name__ == '__main__':
    submission_template_dir = "./data/stage2_sample_submission.csv"
    data_for_prediction = "./data/preprocess_512_stage2"
    out_folder = "./data/unent_prediction"

    # nodule_mask = np.load("./data_luna/out/1.3.6.1.4.1.14519.5.2.1.6279.6001.217754016294471278921686508169_nodule_mask.npz")
    # # print np.mean(nodule_mask, axis=0)
    # print nodule_mask.files
    # print nodule_mask['arr_0'].shape
    all_data = sr.read_luna_csv("./data_luna/CSVFILES/annotations.csv")
    df_train, df_test = sr.split(all_data, 1.0)

    train_generator = sr.luna_unet_gen(df_train, "./data_luna/out")
    test_generator = sr.luna_unet_gen(df_test, "./data_luna/out")
    # =================================== Test generator =============================== #
    # input, target = train_generator.next()                # test generator
    # print input.shape
    # print target.shape
    # np.set_printoptions(threshold='nan')
    # # print target
    # input2, target2 = train_generator.next()  # test generator
    # print target2
    # =================================== Train =============================== #
    # model = unet_model()
    # model.fit_generator(generator=train_generator, steps_per_epoch=1, epochs=600, validation_data=None)
    # model.save('./models/my_model.h5')

    # =================================== Prediction =============================== #
    model = unet_model()
    loaded_model = load_model("./models/unet_magic1.h5", model)
    df_prediction = sr.read_prediction(submission_template_dir)
    predict_nodule_mask(df_prediction, loaded_model, data_for_prediction, out_folder)

    # loaded_model.fit_generator(generator=train_generator, steps_per_epoch=1, epochs=200, validation_data=None)

    # p1 = sr.load_npz("./data/unent_prediction/004828796b994741c4466f59a8c7e9a4.npz")
    # print np.count_nonzero(p1)