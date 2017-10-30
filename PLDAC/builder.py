# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution1D,MaxPooling1D, AveragePooling1D,GlobalMaxPooling1D,GlobalAveragePooling1D
import copy
from keras.wrappers.scikit_learn import BaseWrapper,KerasClassifier
from keras import regularizers
from keras.layers import LSTM,SimpleRNN,GRU
import numpy as np
from preprocess import oneHot

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params


"""
Redefinition des fonctions fit et predict afin de convertir les labels en oneHotVector
Permet d'assurer la compatibilite d'un modele de classification multi-classe avec
les fonctions StratifiedKFold et GridSearchCV
"""

class OneHotKerasClassifier(KerasClassifier):
    
    def fit(self,x,y,**kwargs):
        self.classes = np.unique(y)
         
        if len(self.classes)==2:
            dy=y
        else:
            dy = oneHot(y)
            
        self.history =  super(OneHotKerasClassifier,self).fit(x,dy,**kwargs)
        return self.history
        
    def score(self,x,y,**kwargs):
        if len(self.classes)==2:      
            dy=y
        else:
            dy = oneHot(y)
        return super(OneHotKerasClassifier,self).score(x,dy,**kwargs)      
    
    def evaluate(self,x, y, **kwargs):
        if len(self.classes)==2:      
            dy=y
        else:
            dy = oneHot(y)
        return super(OneHotKerasClassifier,self).evaluate(x,dy,**kwargs)



def MLPBuilder(input_shape,nbClasses,listDims=[10],activation='relu',loss="categorical_crossentropy",opt="adam",listLayerReg = [],regularisations=[],callbacks=None):
    
    dicoReg = dict()
    dicoReg['l1'] = regularizers.l1
    dicoReg['l2'] = regularizers.l2
    dicoLayerReg = dict()
    dicoLayerReg['dropout'] = Dropout
    dicoLayerReg['batchnormalization'] = BatchNormalization
    #Construction du modele
    model = Sequential()
    #Couche d'input
    l = Dense(listDims[0],input_shape=input_shape,activation=activation)
    reg = None
    regularisationIndex = 0
    
    if 0 in listLayerReg :
        reg = regularisations[0][0]
        arg = regularisations[0][1]
        regularisationIndex = (regularisationIndex+1)%(len(regularisations))
        if reg in dicoReg.keys():
            l.activity_regularizer = dicoReg[reg](arg)
            model.add(l)
        elif reg in dicoLayerReg:
            model.add(l)
            model.add(dicoLayerReg[reg](arg))
    else:
        model.add(l)
    
    for layer in range(1,len(listDims)):
        #Ajout de regularisation sur la couche si specifiee        
        l = Dense(listDims[layer],activation=activation)
        reg = None
        if layer in listLayerReg:
            #Construction de l'objet de regularisation
            reg = regularisations[regularisationIndex][0]
            arg = regularisations[regularisationIndex][1]
            regularisationIndex = (regularisationIndex+1)%(len(regularisations))
            if reg in dicoReg.keys():
                l.activity_regularizer = dicoReg[reg](arg)
                model.add(l)
            elif reg in dicoLayerReg:
                model.add(l)
                model.add(dicoLayerReg[reg](arg))
        else:
            model.add(l)
    
    #Couche finale
    if(nbClasses == 2):
        loss = "binary_crossentropy"
        finalActivation = "sigmoid"
    else:
        finalActivation="softmax"
        
    model.add(Dense(nbClasses,activation=finalActivation))
    #Compilation du modele
    model.compile(loss=loss,optimizer = opt,metrics=['accuracy','categorical_accuracy'],callbacks=callbacks)
    model.summary()
    return model


"""
    Fonction permettant de construire un RNN
    
    Parametres:
    
    nbshape : Dimensions des exemples
    nbClasses : Nombre de classes a predire
    listDims : liste contenant le nombre de neurones pour chaque couche
    time: pas de temps
    activations :  fonctions d'activation a utiliser pour chaque couche sauf la dernière
    loss : fonction de cout a utiliser pour notre modele
    opt : optimizer a utiliser (cf documentation Keras pour plus d'infos)
    listLayerReg : liste des couches à regulariser
    regularisations : type de regularisation a utiliser
    callbacks : Liste des callbacks à utiliser dans notre modèle.
    
    Retourne:
    un RNN construit selon les arguments
"""

def SimpleRNNBuilder(nbshape,nbClasses,nb_variation,listDims=[10],activation='relu',loss="categorical_crossentropy",opt="adam",listLayerReg = [],regularisations=[],callbacks=None):
    
    #dictionaire des regularisation
    dicoReg = dict()
    dicoReg['l1'] = regularizers.l1
    dicoReg['l2'] = regularizers.l2
    dicoLayerReg = dict()
    dicoLayerReg['dropout'] = Dropout
    dicoLayerReg['batchnormalization'] = BatchNormalization
    
    
    taille=len(listDims)
    #construction du modele
    model = Sequential()
    drop_batch=0
    if taille==1:
        seq=False
    else:
        seq=True
    M=SimpleRNN(listDims[0],input_shape=(nbshape,nb_variation),activation=activation,return_sequences=seq)
    
    reg=None
    regularisationIndex=0
    
    if 0 in listLayerReg:
        reg=regularisations[0][0]
        arg=regularisations[0][1]
        regularisationIndex+=1
        if reg in dicoReg.keys():
            model.add(M)
            model.layers[0].W_regularizer = dicoReg[reg](arg)
        
        
        elif reg in dicoLayerReg:
            model.add(M)
            model.add(dicoLayerReg[reg](arg))
            drop_batch+=1
    else:
        model.add(M)


    for layer in range(1,len(listDims)):
        #Ajout de regularisation sur la couche si specifiee
        if taille==layer+1:
            seq=False
        else:
            seq=True
        M = SimpleRNN(listDims[layer],activation=activation,return_sequences=seq)
        reg = None
        if layer in listLayerReg:
            #Construction de l'objet de regularisation
            reg = regularisations[regularisationIndex][0]
            arg=regularisations[regularisationIndex][1]
            regularisationIndex = (regularisationIndex+1)%(len(regularisations))
            if reg in dicoReg.keys():
                model.add(M)
                model.layers[layer+drop_batch].W_regularizer=dicoReg[reg](arg)
            
            elif reg in dicoLayerReg:
                model.add(M)
                model.add(dicoLayerReg[reg](arg))
                drop_batch+=1
        else:
            model.add(M)


    #Couche finale
    if(nbClasses == 2):
        loss = "binary_crossentropy"
        finalActivation = "sigmoid"
        nbClasses=1
    else:
        finalActivation="softmax"

    model.add(Dense(nbClasses,activation=finalActivation))
    #Compilation du modele
    model.compile(loss=loss,optimizer = opt,metrics=['categorical_accuracy','accuracy'],callbacks=callbacks)
    model.summary()
    return model

"""
    Fonction permettant de construire un LSTM
    
    Parametres:
    
    nbshape : Dimensions des exemples
    nbClasses : Nombre de classes a predire
    time: pas de temps
    listDims : liste contenant le nombre de neurones pour chaque couche
    activations :  fonctions d'activation a utiliser pour chaque couche sauf la dernière
    loss : fonction de cout a utiliser pour notre modele
    opt : optimizer a utiliser (cf documentation Keras pour plus d'infos)
    listLayerReg : liste des couches à regulariser
    regularisations : type de regularisation a utiliser
    callbacks : Liste des callbacks à utiliser dans notre modèle.
    
    Retourne:
    un LSTM construit selon les arguments
    """

def LSTMBuilder(nbshape,nbClasses,nb_variation,listDims=[10],activation='relu',loss="categorical_crossentropy",opt="adam",listLayerReg = [],regularisations=[],callbacks=None):
    dicoReg = dict()
    dicoReg['l1'] = regularizers.l1
    dicoReg['l2'] = regularizers.l2
    dicoLayerReg = dict()
    dicoLayerReg['dropout'] = Dropout
    dicoLayerReg['batchnormalization'] = BatchNormalization
    #Construction du modele
    
    taille=len(listDims)
    model = Sequential()
    drop_batch=0
    if taille==1:
        seq=False
    else:
        seq=True
    M=LSTM(listDims[0],input_shape=(nbshape,nb_variation),activation=activation,return_sequences=seq,unroll=True)
    
    reg=None
    regularisationIndex=0
    if 0 in listLayerReg:
        reg=regularisations[0][0]
        arg=regularisations[0][1]
        regularisationIndex+=1
        if reg in dicoReg.keys():
            model.add(M)
            model.layers[0].W_regularizer = dicoReg[reg](arg)
        
        
        elif reg in dicoLayerReg:
            model.add(M)
            model.add(dicoLayerReg[reg](arg))
            drop_batch+=1
    else:
        model.add(M)


    for layer in range(1,len(listDims)):
        #Ajout de regularisation sur la couche si specifiee
        if taille==layer+1:
            seq=False
        else:
            seq=True
        M = LSTM(listDims[layer],activation=activation,return_sequences=seq)
        reg = None
        if layer in listLayerReg:
            #Construction de l'objet de regularisation
            reg = regularisations[regularisationIndex][0]
            arg=regularisations[regularisationIndex][1]
            regularisationIndex = (regularisationIndex+1)%(len(regularisations))
            if reg in dicoReg.keys():
                model.add(M)
                model.layers[layer+drop_batch].W_regularizer=dicoReg[reg](arg)
            
            elif reg in dicoLayerReg:
                model.add(M)
                model.add(dicoLayerReg[reg](arg))
                drop_batch+=1
        else:
            model.add(M)


    #Couche finale
    if(nbClasses == 2):
        loss = "binary_crossentropy"
        finalActivation = "sigmoid"
        nbClasses=1
    else:
        finalActivation="softmax"

    model.add(Dense(nbClasses,activation=finalActivation))
    #Compilation du modele
    model.compile(loss=loss,optimizer = opt,metrics=['categorical_accuracy','accuracy'],callbacks=callbacks)
    model.summary()
    return model



"""
    Fonction permettant de construire un GRU
    
    Parametres:
    
    nbshape : Dimensions des exemples
    nbClasses : Nombre de classes a predire
    time: pas de temps
    listDims : liste contenant le nombre de neurones pour chaque couche
    activations :  fonctions d'activation a utiliser pour chaque couche sauf la dernière
    loss : fonction de cout a utiliser pour notre modele
    opt : optimizer a utiliser (cf documentation Keras pour plus d'infos)
    listLayerReg : liste des couches à regulariser
    regularisations : type de regularisation a utiliser
    callbacks : Liste des callbacks à utiliser dans notre modèle.
    
    Retourne:
    un GRU construit selon les arguments
    """

def GRUBuilder(nbshape,nbClasses,nb_variation,listDims=[10],activation='relu',loss="categorical_crossentropy",opt="adam",listLayerReg = [],regularisations=[],callbacks=None):
    dicoReg = dict()
    dicoReg['l1'] = regularizers.l1
    dicoReg['l2'] = regularizers.l2
    dicoLayerReg = dict()
    dicoLayerReg['dropout'] = Dropout
    dicoLayerReg['batchnormalization'] = BatchNormalization
    #Construction du modele
    
    taille=len(listDims)
    drop_batch=0
    model = Sequential()
    if taille==1:
        seq=False
    else:
        seq=True
    M=GRU(listDims[0],input_shape=(nbshape,nb_variation),activation=activation,return_sequences=seq)
    
    reg=None
    regularisationIndex=0
    if 0 in listLayerReg:
        reg=regularisations[0][0]
        arg=regularisations[0][1]
        regularisationIndex+=1
        if reg in dicoReg.keys():
            model.add(M)
            model.layers[0].W_regularizer = dicoReg[reg](arg)
        
        
        elif reg in dicoLayerReg:
            model.add(M)
            model.add(dicoLayerReg[reg](arg))
            drop_batch+=1
    else:
        model.add(M)
    for layer in range(1,len(listDims)):
        #Ajout de regularisation sur la couche si specifiee
        if taille==layer+1:
            seq=False
        else:
            seq=True
        M = GRU(listDims[layer],activation=activation,return_sequences=seq)
        reg = None
        if layer in listLayerReg:
            #Construction de l'objet de regularisation
            reg = regularisations[regularisationIndex][0]
            arg=regularisations[regularisationIndex][1]
            
            regularisationIndex = (regularisationIndex+1)%(len(regularisations))

            if reg in dicoReg.keys():
                model.add(M)
                model.layers[layer+drop_batch].W_regularizer=dicoReg[reg](arg)
            
            elif reg in dicoLayerReg:
                model.add(M)
                model.add(dicoLayerReg[reg](arg))
                drop_batch+=1
        else:
            model.add(M)


    #Couche finale
    if(nbClasses == 2):
        loss = "binary_crossentropy"
        finalActivation = "sigmoid"
        nbClasses=1
    else:
        finalActivation="softmax"

    model.add(Dense(nbClasses,activation=finalActivation))
    #Compilation du modele
    model.compile(loss=loss,optimizer = opt,metrics=['categorical_accuracy','accuracy'],callbacks=callbacks)
    model.summary()
    return model





"""
    Fonction permettant de construire un CNN
    
    Parametres:
    
    nbshape : shape de l'exemple
    list_nb_filter: liste de filtre par couche
    nbClasses : Nombre de classes a predire
    list_filter_length: taille du filtre par couche
    list_max_pooling: liste pooling par couche
    activations : liste des fonctions d'activation a utiliser pour chaque couche
    loss : fonction de cout a utiliser pour notre modele
    opt : optimizer a utiliser (cf documentation Keras pour plus d'infos)
    listLayerReg : liste des couches à regulariser
    regularisations : type de regularisation a utiliser
    callbacks : Liste des callbacks à utiliser dans notre modèle.
    
    Retourne:
    un CNN construit selon les arguments
    """


def CNNBuilder(nbshape,list_nb_filter,nbClasses,list_filter_length,list_pooling,activation='relu',loss="categorical_crossentropy",opt="adam",listLayerReg = [],regularisations=[],callbacks=None):
    
    dicoReg = dict()
    dicoReg['l1'] = regularizers.l1
    dicoReg['l2'] = regularizers.l2
    dicoLayerReg = dict()
    dicoLayerReg['dropout'] = Dropout
    dicoLayerReg['batchnormalization'] = BatchNormalization
    dicoPool=dict()
    dicoPool['avg'] = AveragePooling1D
    dicoPool['max'] = MaxPooling1D
    dicoPool['globalavg'] = GlobalAveragePooling1D
    dicoPool['globalmax'] = GlobalMaxPooling1D
    
    model=Sequential()
    regularisationIndex = 0
    M=Convolution1D(nb_filter=list_nb_filter[0], filter_length=list_filter_length[0], activation=activation, input_shape=(nbshape,1))
    reg=None
    if 0 in listLayerReg :
        reg = regularisations[0][0]
        arg = regularisations[0][1]
        regularisationIndex = (regularisationIndex+1)%(len(regularisations))
        if reg in dicoReg.keys():
            M.activity_regularizer = dicoReg[reg](arg)
            model.add(M)
        elif reg in dicoLayerReg:
            model.add(M)
            model.add(dicoLayerReg[reg](arg))
    else:
        model.add(M)
    
    name=list_pooling[0][0]
    val=list_pooling[0][1]
    if name=='avg' or name=='max':
        model.add(dicoPool[name](val))
    else:
        model.add(dicoPool[name]())


    for layer in range(1,len(list_nb_filter)):
        #Ajout de regularisation sur la couche si specifiee
        M = Convolution1D(list_nb_filter[layer],list_filter_length[layer])
        reg = None
        if layer in listLayerReg:
            #Construction de l'objet de regularisation
            reg = regularisations[regularisationIndex][0]
            arg = regularisations[regularisationIndex][1]
            regularisationIndex = (regularisationIndex+1)%(len(regularisations))
            if reg in dicoReg.keys():
                M.activity_regularizer = dicoReg[reg](arg)
                model.add(M)
            elif reg in dicoLayerReg:
                model.add(M)
                model.add(dicoLayerReg[reg](arg))
        else:
            model.add(M)


        name=list_pooling[layer][0]
        val=list_pooling[layer][1]
        if name=='avg' or name=='max':
            model.add(dicoPool[name](val))
        else:
            model.add(dicoPool[name]())

    if(nbClasses == 2):
        loss = "binary_crossentropy"
        finalActivation = "sigmoid"
        classe=1
    else:
        finalActivation="softmax"
        classe=nbClasses

    model.add(Flatten())
    model.add(Dense(classe,activation=finalActivation))
    #Compilation du modele
    model.compile(loss=loss,optimizer = opt,metrics=['accuracy','categorical_accuracy'],callbacks=callbacks)
    model.summary()
    return model

