# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:19:36 2017

@author: akli
"""
from sklearn.model_selection import StratifiedKFold
import KerasGridSearcher
from keras.wrappers.scikit_learn import BaseWrapper
import copy
import pandas as pd
from preprocess import oneHot

"""
Necessaire a la compatibilite sklearn et keras
"""

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params

"""
Permet d'effectuer un gridsearch et de recuperer un objet grid_result donnant tous les resultats

    Arguments :
        datax : donnees a traiter
        datay : labels/classes de chaque donnee
        buildfn : fonction retourner un classifier selon les arguments du dictionnaire dictParams
        dictParams : dictionnaire avec comme entree le nom du parametre et en valeur la liste de ses differentes valeurs
        n_splits : nombre de sous-groupes a utiliser en cross-validation
        binary : vrai si la classification est binaire, faux sinon
        
    Retourne : 
        Objet grid_result contenant le resultat de toutes les combinaisons des parametres
"""

def GridSearch(model,datax,datay,dictParams,n_splits=3,binary=False):
    """
    if(not binary):
        model = OneHotKerasClassifier(buildfn=buildfn)
    else:
        model = KerasClassifier(buildfn=buildfn)
    """
    #Division du datasets en n_splits+1 parties, la partie en plus est celle de validation
    skf = StratifiedKFold(n_splits = n_splits+1) 
    #Generateur donnant les index des sous-groupes des donnees
    gen = skf.split(datax,datay)
    #Recuperation des indices des exemples du validation set
    valIndex = next(gen)[1]
    #Ensemble des indices
    indexes = range(len(datax))
    #Recuperation du set de validation
    valData = datax[valIndex]
    valLabels = datay[valIndex]
    #Reste des donnees 
    remainingData = datax[[j for j in indexes if j not in valIndex]]
    remainingLabels = datay[[j for j in indexes if j not in valIndex]]
    #Ajout du set de validation aux parametres du dictionnaire
    if not binary:
        dictParams["validation_data"] = [(valData,oneHot(valLabels))]
    else :
        dictParams["validation_data"] = [(valData,valLabels)]
    #Nouveau split
    skf.n_splits = n_splits
    gen = skf.split(remainingData,remainingLabels)
    #GridSearch 
    grid = KerasGridSearcher.GridSearchCV(estimator = model,param_grid=dictParams,cv=gen,n_jobs=-1)
    return grid.fit(remainingData,remainingLabels)

"""
Permet d'enregistrer les resultats du gridSearch en format HTML
    Arguments:
        grid_result : resultats du GridSearchCV tel que donne par la fonction sklearn
        name : nom du fichier HTML dans lequel les resultats sont enregistrés
"""
def SaveGridSearchResult(grid_result,name):
    res = grid_result.cv_results_
    df = pd.DataFrame.from_dict(res).drop("params",1).sort('rank_test_score')
    df.to_html(name+".html")
    

