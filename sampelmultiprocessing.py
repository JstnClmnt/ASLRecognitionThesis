import pandas as pd
import numpy as np
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
import os
import math
import time
import multiprocessing
from random import shuffle
np.seterr(invalid='ignore')

def cross_validate(word):
    data=pd.read_csv("dataset1-6.csv")
    true=0
    traininglength=0
    testlength=0
    dataperword=data[data["gesture"]==word]
    speakers=dataperword["speaker"].unique().tolist()
    shuffle(speakers)
    datalength=len(speakers)
    for index in range(datalength):
        trainingdata=data[data["gesture"]!=word]
        trainingdata=trainingdata.append(dataperword[dataperword["speaker"]!=speakers[index]])
        testdata=dataperword[dataperword["speaker"]==speakers[index]]
        training=train_all(trainingdata)
        print(training)
        print("Printing HMMs for the word: "+word+"; Test is "+speakers[index])
        guessword=recognize(training,dataperword[dataperword["speaker"]==speakers[index]])
        if(guessword==word):
            true=true+1
    print("Recognized "+word+":"+ str(true)+" out of "+str(len(speakers)))
    #print("Total Data: "+str(datalength)+" Training Set: "+str(0)+"-"+str(traininglength)+" Test Set:"+str(datalength-testlength)+"-"+str(datalength-1))
    return [word,true,len(speakers)]
def train_all(df):
    models={}
    words=df["gesture"].unique()
    for word in words:
        dataword=df[df["gesture"]==word]
        speakers=dataword["speaker"].unique()
        lengths=[]
        for speaker in speakers:
            lengths.append(len(dataword[dataword["speaker"]==speaker]))
        dataword=dataword.drop(columns=[dataword.columns[56],dataword.columns[57],dataword.columns[58]])
        #BAYESIAN INFORMATION CRITERION FOR SELECTING THE BEST MODEL
        best_score,best_model=float("inf"),None
        models[word]=GaussianHMM(n_components=2,covariance_type="diag", n_iter=1000).fit(dataword,lengths)
        for x in range(2,30):
            try:
                model=GaussianHMM(n_components=x,covariance_type="diag", n_iter=1000).fit(dataword,lengths)
                logL=model.score(dataword,lengths)
                n_features=dataword.shape[1]
                n_params =x * (x - 1) + 2 * n_features * x
                logN = np.log(dataword.shape[0])
                bic = -2 * logL + n_params * logN
                if bic < best_score:
                    best_score, best_model = bic, model
            except Exception as e:
                break
        models[word]=best_model
    return models

def recognize(dictModels,data):
    probmodels={}
    prob=float("-inf")
    word=None
    data=data.drop(columns=[data.columns[56],data.columns[57],data.columns[58]])
    for modelword,model in dictModels.items():
        try:
            modelprob=model.decode(data)[0]
            probmodels[modelword]=modelprob
            if(modelprob>prob):
                prob=modelprob
                word=modelword
        except Exception as e:
            continue

    print(probmodels)
    return word
if __name__ == '__main__':
    data=pd.read_csv("dataset1-6.csv")
    data=data.replace(0.0,float("inf"))
    gestures=data["gesture"].unique()
    #data=data[data["speaker"]!="ling_f"]
    for x in gestures:
        datapergesture=data[data["gesture"]==x]
        lenSpeakers=len(datapergesture["speaker"].unique())
    words=data["gesture"].unique().tolist()
    totalTrue=0
    totalTest=0
    starttime = time.time()
    pool = multiprocessing.Pool()
    print(pool.map(cross_validate,words))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))


