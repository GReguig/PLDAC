#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:48:39 2017

@author: meliss
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Reshape
from keras import regularizers
from keras.layers import LSTM
from keras import backend as K
from keras.utils import np_utils
import matplotlib.image as mpimg
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils


##extraction de donnees: univarié
"""
    params:
    fichier: un fichier h5py
    dataset_name: le nom du dataset
    """
def Test_Train(fichier, dataset_name):
    file    = h5py.File(fichier, 'r')   
    dataset = file[dataset_name]
    
    #creation fichier test
    Test  = dataset['TEST']
    label_test=Test['labels']
    static_test=Test['static']
    ts_test=Test['ts']
    
    #label de test
    label=label_test['values']
    #value de test
    static0=static_test['axis0']
    static1=static_test['axis1']
    ts_axis0=ts_test['axis0']
    ts_axis1=ts_test['axis1']
    ts_axis2=ts_test['axis2']
    ts_block0_items=ts_test['block0_items']
    ts_block0_values=ts_test['block0_values']
    
    #recuperation des data test
    X_Test=ts_block0_values.value
    Y_Test=label.value
    
    #creation fichier train 
    Train  = dataset['TRAIN']
    label_train=Train['labels']
    static_train=Train['static']
    ts_train=Train['ts']
    #label de test
    label1=label_train['values']
    #value de test
    static01=static_train['axis0']
    static11=static_train['axis1']
    ts_axis01=ts_train['axis0']
    ts_axis11=ts_train['axis1']
    ts_axis21=ts_train['axis2']
    ts_block0_items1=ts_train['block0_items']
    ts_block0_values1=ts_train['block0_values']
    
    #recuperation des data train
    X_Train=ts_block0_values1.value
    Y_Train=label1.value
    """
    ## re shape des donnes Y en 2 dimensions
    if 0 in np.unique(Y_Train):
        Y_Train=np_utils.to_categorical(Y_Train)
        Y_Test=np_utils.to_categorical(Y_Test)
    else:
        Y_Train=np_utils.to_categorical(Y_Train)[:,1:]
        Y_Test=np_utils.to_categorical(Y_Test)[:,1:]
    """
    return X_Test,Y_Test,X_Train,Y_Train
   

#extraction de donnees pour chaque univarié
"""
fichier = 'TimeSeriesData/dataUCR_nov2015.h5'
dataset_name   = ['Beef','ECGFiveDays','CBF','ArrowHead','ItalyPowerDemand','MedicalImages']

X_Beef_TEST,Y_Beef_TEST,X_Beef_TRAIN,Y_Beef_TRAIN=Test_Train(fichier,dataset_name[0])


X_ECG_TRAIN,Y_ECG_TRAIN,X_ECG_TEST,Y_ECG_TEST=Test_Train(fichier,dataset_name[1])


X_CBF_TRAIN,Y_CBF_TRAIN,X_CBF_TEST,Y_CBF_TEST=Test_Train(fichier,dataset_name[2])


X_AH_TRAIN,Y_AH_TRAIN,X_AH_TEST,Y_AH_TEST=Test_Train(fichier,dataset_name[3])


X_ITALY_TRAIN,Y_ITALY_TRAIN,X_ITALY_TEST,Y_ITALY_TEST=Test_Train(fichier,dataset_name[4])

X_MEDI_TRAIN,Y_MEDI_TRAIN,X_MEDI_TEST,Y_MEDI_TEST=Test_Train(fichier,dataset_name[5])
"""