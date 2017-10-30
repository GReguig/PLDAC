# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:12:31 2017

@author: 3303535
"""
from keras.models import Sequential
from keras.callbacks import History,EarlyStopping,TensorBoard
from preprocess import oneHot
import numpy as np 
import modelBuilder
from keras.layers import Dense,Activation,Dropout,Flatten,Reshape
import gridSearch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,ParameterGrid
import KerasGridSearcher


#On récupère les fichiers textes sous forme d'une array
train = np.genfromtxt("../../Beef/Beef_TRAIN",dtype=float,delimiter=",")
test = np.genfromtxt("../../Beef/Beef_TEST",dtype=float,delimiter=",")
datax = np.concatenate((train[:,1:],test[:,1:]))
labels = np.concatenate((train[:,0],test[:,0]))

"""
for c in nbCouches:
    l = [c * [nbN] for nbN in nbNeurones]
    listeDimensions = listeDimensions + l
""" 
dico = dict()
dico['input_shape'] = [datax.shape[1:]]
dico['nbClasses'] = [len(np.unique(labels))]
dico["listDims"] = [[10],[15,2],[3]]
#dico["batch_size"] = [10,20,30,40,50]
dico['listLayerReg'] = [[0]]
valsReg = [0.01,0.1,1]
dico['regularisations'] = [[("l1",[0.2])]]


earlyStop = EarlyStopping(monitor='val_loss',min_delta =0.1,patience=30,mode='min', verbose=2)
board = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

"""
model = Sequential()
model.add(Dense(10,input_dim = datax.shape[1]))
model.add(Dense(5))
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
model.history = model.fit(datax,oneHot(labels),validation_split=0.4,epochs=2000,callbacks=[earlyStop,board])
"""
model = modelBuilder.OneHotKerasClassifier(build_fn = modelBuilder.MLPBuilder,nb_epoch=500,verbose = 2)
"""
skf = StratifiedKFold(n_splits = 4)

#Generateur donnant les index des sous-groupes des donnees
gen = skf.split(datax,labels)
for i in gen:
    print(i[1].shape)
    
print("Fin premier Gen")
skf.n_splits = 2
gen = skf.split(datax,labels)
for i in gen :
    print(i)
valIndex = next(gen)[1]
indexes = range(len(datax))
valData = datax[valIndex]
valLabels = labels[valIndex]
remainingData = datax[[j for j in indexes if j not in valIndex]]
remainingLabels = labels[[j for j in indexes if j not in valIndex]]
#GridSearch 
#grid = GridSearchCV(estimator = model,param_grid=dictParams,cv=gen,n_jobs=-1)
"""
grid_result = gridSearch.GridSearch(model,datax,labels,dico)

res = grid_result.cv_results_
"""
df = pd.DataFrame.from_dict(res).drop("params",1).sort('rank_test_score')
df.to_html("test.html")

a = df[['param_listDims','mean_test_score','std_test_score','mean_train_score','std_train_score']]
a.to_html("syntheseMLPl1.html")
nbCouches = range(1,11)
nbNeurones = [16,32,64,128,256,512]
moyTest = []
moyTrain = []
stdTrain = []
stdTest = []
for i in nbCouches: 
    #plt.figure()
    tmpDF = a[a['Nb_Couches'] == i]
    #tmpDF = tmpDF.sort('Nb_Couches')
    moyTrain.append(tmpDF['mean_train_score'].mean())
    moyTest.append(tmpDF['mean_test_score'].mean())
    stdTrain.append(tmpDF['std_train_score'].mean())
    stdTest.append(tmpDF["std_test_score"].mean())
    plt.errorbar(tmpDF['Nb_Couches'],tmpDF['mean_test_score'],yerr=tmpDF['std_test_score'],fmt='-o')
    plt.errorbar(tmpDF['Nb_Couches'],tmpDF['mean_train_score'],yerr = tmpDF['std_train_score'],fmt='-o')
    plt.ylim((0.0,1.1))
    plt.xlim((-0.5,11))
    plt.xlabel("Nombre de couches cachees")
    plt.ylabel("Accuracy")
    plt.legend(["Score en test","Score en train"],loc=3,borderaxespad=0.)
    plt.title("Evolution de l'accuracy en train/test pour des couches a %d neurones"%(i))
    plt.savefig("MLPNeurones/MLP%d.png"%(i)) 
plt.figure()
plt.errorbar(nbCouches,moyTest,yerr=stdTest,fmt='-o')
plt.errorbar(nbCouches,moyTrain,yerr=stdTrain,fmt='-o')
plt.ylim((0.0,1.1))
plt.xlim((-0.5,11))
plt.xlabel("Nombre de couches")
plt.ylabel("Accuracy")
plt.legend(['Score en test','Score en train'],loc = 3)
plt.title("Performances moyennes des modeles selon le nombre de couches")
plt.savefig("MLPCouches/PerfMoy.png")
nbCouches = []
nbNeurones = []

for i in a['param_listDims'] : 
    nbNeurones.append(i[0])
    nbCouches.append(len(i))

a["Nb_Couches"] = pd.Series(nbCouches)
a["Nb_Neurones/Couche"] = pd.Series(nbNeurones)
a = a.drop("param_listDims",1)
"""
