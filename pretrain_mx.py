import numpy as np
import dicom
import os
import cv2
import mxnet as mx
import pandas as pd

def calc_features(pretrained_model, patient_id, patient_data, out_folder):
    batch = get_data_id(patient_data)
    feats = pretrained_model.predict(batch)
    print("INFO: Predict feature using resnet-50 for patient: %s" % patient_id)
    print(feats)
    print("INFO: Predict feature has shape " + str(feats.shape))
    out_folder += "/" + patient_id
    np.save(out_folder, feats)

		
def get_extractor():
    model = mx.model.FeedForward.load('model/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor

	
def get_data_id(patient_data):
    batch = []
    for i in range(0, patient_data.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = patient_data[i + j]
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))
    batch = np.array(batch)
    return batch	
