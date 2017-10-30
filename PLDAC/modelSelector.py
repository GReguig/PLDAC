#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:56:00 2017

@author: akli
"""

import generateParams
import builder
import gridSearch
import numpy as np

def modelSelector(datax,datay,nomData):
    #Reshape pour les CNN et les RNN
    dx = datax.reshape((datax.shape[0],datax.shape[1],1))
    #Dictionnaires des parametres
    dicoMLP = generateParams.generateDicoMLP(datax,datay)
    dicoCNN = generateParams.generateDicoCNN(dx,datay)
    dicoRNN = generateParams.generateDicoRNN(dx,datay)
    dicoGRU = generateParams.generateDicoGRU(dx,datay)
    #Modeles
    modelMLP = builder.OneHotKerasClassifier(build_fn = builder.MLPBuilder,verbose = 1)
    modelCNN = builder.OneHotKerasClassifier(build_fn = builder.CNNBuilder,verbose = 1)
    modelSimpleRNN = builder.OneHotKerasClassifier(build_fn = builder.SimpleRNNBuilder,verbose=1)
    modelGRU = builder.OneHotKerasClassifier(build_fn = builder.GRUBuilder,verbose=1)
    #GridSearches
    #CNN
    """
    print("GridSearch sur les CNN en cours")
    grid_resultCNN = gridSearch.GridSearch(modelCNN,dx,datay,dicoCNN)
    gridSearch.SaveGridSearchResult(grid_resultCNN,"Resultats/%sCNN"%(nomData))
    print("GridSearch CNN fini, resultats dans : Resultats/%sCNN"%(nomData))
    #MLP
    print("GridSearch sur les MLP en cours")
    grid_resultMLP = gridSearch.GridSearch(modelMLP,datax,datay,dicoMLP)
    gridSearch.SaveGridSearchResult(grid_resultMLP,"Resultats/%sMLP"%(nomData))
    print("GridSearch MLP fini, resultats dans : Resultats/%sMLP"%(nomData))
	
	#SimpleRNN
	print("GridSearch sur les SimpleRNN en cours")
    grid_resultRNN = gridSearch.GridSearch(modelSimpleRNN,dx,datay,dicoRNN)
    gridSearch.SaveGridSearchResult(grid_resultRNN,"Resultats/%sRNN"%(nomData))
    print("GridSearch SimpleRNN fini, resultats dans : Resultats/%sRNN"%(nomData))
	#GRU
    """
    print("GridSearch sur les GRU en cours")
    grid_resultGRU = gridSearch.GridSearch(modelGRU,dx,datay,dicoGRU)
    gridSearch.SaveGridSearchResult(grid_resultGRU,"Resultats/%sGRU"%(nomData))
    print("GridSearch GRU fini, resultats dans : Resultats/%sGRU"%(nomData))

dx = np.load("Data/Adiac/X.npy")
dy = np.load("Data/Adiac/Y.npy")

modelSelector(dx,dy,"Adiac")
