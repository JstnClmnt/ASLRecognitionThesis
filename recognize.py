import pandas as pd
import numpy as np
from hmmlearn import hmm
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
import operator
import joblib
import os
def recognize(dictModels,data,n_components):
    probmodels={}
    prob=float("-inf")
    data=data.drop(columns=[data.columns[56],data.columns[57],data.columns[58]])
    #data=(data-data.min())/(data.max()-data.min())
    #data=data.fillna(0.0)
    x=n_components
    for modelword,model in dictModels.items():
        try:
            logL=model.score(data)
            n_features=data.shape[1]
            n_params =x * (x - 1) + 2 * n_features * x
            logN = np.log(data.shape[0])
            bic = -2 * logL + n_params * logN
            #modelprob=model.score(data)
            probmodels[modelword]=bic
            if(bic>prob):
                prob=bic
                word=modelword
        except Exception as e:
            probmodels[modelword]=float("-inf")
            word=modelword
            continue
    print(sorted(probmodels.items(), key=operator.itemgetter(1),reverse=True))
    #print(probmodels)
    #print(sorted(probmodels.items(), key=operator.itemgetter(1),reverse=True)[0][0])
    return sorted(probmodels.items(), key=operator.itemgetter(1),reverse=True)[0][0]

models={}
path = 'HMMmodels/fingerspelling'
testData=pd.read_csv("test.csv")
for filename in os.listdir(path):
    #print("HMMmodels/dynamicset1/"+filename)
    models[filename.split(".pkl")[0]]=joblib.load("HMMmodels/fingerspelling/"+filename)

print(recognize(models,testData,21))
    