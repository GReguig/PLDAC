# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:41:38 2017

@author: 3303535
"""

import numpy as np
from keras.callbacks import EarlyStopping

"""
Genere un dictionnaire de parametres pour effectuer un gridSearch
Compatible avec MLPBuilder
"""
def generateDicoMLP(datax,datay):
    #Dictionnaire des parametres
    params = dict()
    #On fixe le nombre d'epochs a 500
    params['epochs'] = [500]
    #Dimensions des donnees d'entree
    params['input_shape'] = [datax.shape[1:]]
    #Nombre de classes
    params['nbClasses'] = [len(np.unique(datay))]
    #Liste des dimensions de chaque couche (nbNeurones + nbCouches)
    params['listDims'] = [np.random.randint(10,100,size=nbLayers) for nbLayers in range(1,4,1)]
    #Couches de regularisation
    params['listLayerReg'] = [[0],[0,2],[1]]
    #Regularisations
    params['regularisations'] = [[['l2',0.1],['l2',0.01],['dropout',0.5]],[['dropout',0.5]],[['l2',0.1]],[['l2',0.01]]]
    #Callbacks
    callbacks=[]
    #Arret de l'apprentissage si plateau
    earlystopping = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=40,mode='min',verbose=2)
    #Sauvegarde du meilleur modele en fonction de loss en validation
    #modelCheckpoint = ModelCheckpoint('Best_MLP',save_best_only=True)    
    callbacks.append(earlystopping)
    #callbacks.append(modelCheckpoint)
    params['callbacks'] = [callbacks]
    return params
    
    
"""
"""
def generateDicoCNN(datax,datay):
    params = dict()
    params['epochs'] = [500]
    params['nbshape'] = [datax.shape[1]]
    params['nbClasses'] = [len(np.unique(datay))]
    #params['list_nb_filter'] = [np.random.randint(5,30,size=nbLayers) for nbLayers in range(1,4,1)]
    params['list_nb_filter'] = [[10]]
    params['list_filter_length'] = [[3,3,3],[5,5,5]]
    params['list_pooling'] = [[['max',3],['max',3],['max',3]],[['max',5],['max',3],['max',1]]]
    params['listLayerReg'] = [[0],[0,2],[1]]
    params['regularisations'] = [[['l2',0.1],['l2',0.01],['dropout',0.5]],[['dropout',0.5]],[['l2',0.1]],[['l2',0.01]]]
    callbacks=[]
    #Arret de l'apprentissage si plateau
    earlystopping = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=40,mode='min',verbose=2)
    #Sauvegarde du meilleur modele en fonction de loss en validation
    #modelCheckpoint = ModelCheckpoint('Best_MLP',save_best_only=True)    
    callbacks.append(earlystopping)
    #callbacks.append(modelCheckpoint)
    params['callbacks'] = [callbacks]
    return params
    
def generateDicoRNN(datax,datay):
    #Dictionnaire des parametres
    params = dict()
    #On fixe le nombre d'epochs a 500
    params['epochs'] = [500]
    #loss
    params['loss'] = [['categorical_crossentropy']]
    #Dimensions des donnees d'entree
    params['nbshape'] = [datax.shape[1]]
    #Dimensions de variation univarié ou multivarié
    params['nb_variation'] = [1]
    #Dimensions des donnees d'entree
    params['nbshape'] = [datax.shape[1]]
    #Nombre de classes
    params['nbClasses'] = [len(np.unique(datay))]
    #Liste des dimensions de chaque couche (nbNeurones + nbCouches)
    params['listDims'] = [np.random.randint(1,10,size=nbLayers) for nbLayers in range(1,3,1)]
    #Couches de regularisation
    params['listLayerReg'] = [[0],[],[0,1]]
    #Regularisations
    params['regularisations'] = [[['l2',0.1],['dropout',0.5]],[['dropout',0.3]],[['l2',0.01]],[['l2',0.1]]]
    #Callbacks
    callbacks=[]
    #Arret de l'apprentissage si plateau
    earlystopping = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=40,mode='min',verbose=2)
    #Sauvegarde du meilleur modele en fonction de loss en validation
    #modelCheckpoint = ModelCheckpoint('Best_MLP',save_best_only=True)    
    callbacks.append(earlystopping)
    #callbacks.append(modelCheckpoint)
    params['callbacks'] = [callbacks]
    return params

def generateDicoGRU(datax,datay):
    #Dictionnaire des parametres
    params = dict()
    #On fixe le nombre d'epochs a 500
    params['epochs'] = [500]
    #Dimensions des donnees d'entree
    params['nbshape'] = [datax.shape[1]]
    #Nombre de classes
    params['nbClasses'] = [len(np.unique(datay))]
    params['nb_variation'] = [1]
    #Liste des dimensions de chaque couche (nbNeurones + nbCouches)
    params['listDims'] = [np.random.randint(1,10,size=nbLayers) for nbLayers in range(1,3,1)]
    #Couches de regularisation
    params['listLayerReg'] = [[],[0],[1],[0,1]]
    #Regularisations
    params['regularisations'] = [[['l2',0.1],['l2',0.01],['dropout',0.5]],[['dropout',0.5]],[['l2',0.01]],[['l2',0.1]]]
    #Callbacks
    callbacks=[]
    #Arret de l'apprentissage si plateau
    earlystopping = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=40,mode='min',verbose=2)
    #Sauvegarde du meilleur modele en fonction de loss en validation
    #modelCheckpoint = ModelCheckpoint('Best_GRU',save_best_only=True)    
    callbacks.append(earlystopping)
    #callbacks.append(modelCheckpoint)
    params['callbacks'] = [callbacks]
    return params
