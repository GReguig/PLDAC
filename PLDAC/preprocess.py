# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:47:02 2017

@author: akli
"""
import numpy as np

"""
Permet de transformer un vecteur de labels en un ensemble de vecteurs oneHot
    Argument
        Y : array-like des labels
    Retourne
        Un ensemble de vecteurs de forme oneHot
"""

def oneHot(Y):
    #Liste des différentes classes
    classes = np.unique(Y)
    #Reshape des exemples, traitement des liste en array
    Y = np.asarray(Y).reshape((-1,1))
    #Nouvelle matrice des labels codées en onehot
    onehot = np.zeros((len(Y),len(classes)))
    
    for i in range(len(classes)):
        #Liste des index des exemples de classe classes[i]
        tmp = np.where(Y==classes[i])[0]
        onehot[tmp,i] = 1
    return onehot
