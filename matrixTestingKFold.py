import pandas as pd
import numpy as np
from hmmlearn import hmm
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import math
import time
import multiprocessing
from random import shuffle
import operator
import multiprocessing
from functools import partial
from sklearn.model_selection import KFold
np.seterr(invalid='ignore')

def recognize(dictModels,data,n_components):
    probmodels={}
    prob=float("-inf")
    word=None
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
    #print(sorted(probmodels.items(), key=operator.itemgetter(1),reverse=True))
    #print(probmodels)
    #print(sorted(probmodels.items(), key=operator.itemgetter(1),reverse=True)[0][0])
    return sorted(probmodels.items(), key=operator.itemgetter(1),reverse=True)[0][0]


def train_all(df,n_components):
    models={}
    words=df["gesture"].unique().tolist()
    for word in words:
        dataword=df[df["gesture"]==word]
        speakers=dataword["speaker"].unique()
        lengths=[]
        for speaker in speakers:
            lengths.append(len(dataword[dataword["speaker"]==speaker]))
        dataword=dataword.drop(columns=[dataword.columns[56],dataword.columns[57],dataword.columns[58]])
        #dataword=(dataword-dataword.min())/(dataword.max()-dataword.min())
        #dataword=dataword.fillna(0.0)
        #BAYESIAN INFORMATION CRITERION FOR SELECTING THE BEST MODEL
        '''best_score,best_model=float("inf"),None
        models[word]=GaussianHMM(n_components=2,covariance_type="diag", n_iter=1000).fit(dataword,lengths)
        for x in range(2,20):
            try:
                model=GaussianHMM(n_components=x,covariance_type="diag", n_iter=1000).fit(dataword,lengths)
                logL=models[word].score(dataword,lengths)
                n_features=dataword.shape[1]
                n_params =x * (x - 1) + 2 * n_features * x
                logN = np.log(self.X.shape[0])
                bic = -2 * logL + n_params * logN
                if bic < best_score:
                    best_score, best_model = bic, model
            except Exception as e:
                continue'''
        models[word]=GaussianHMM(n_components=n_components,covariance_type="spherical", n_iter=1000).fit(dataword,list(lengths))
        #print(models)
    return models
def testing(x,word):
    #print(x)
    data=pd.read_csv("datacsv/datasetcomplete.csv")
    true=0
    traininglength=0
    testlength=0
    dataperword=data[data["gesture"]==word]
    speakers=dataperword["speaker"].unique().tolist()
    datalength=len(speakers)
    '''if(datalength%2==0):
        traininglength=math.floor((datalength*.60))
        testlength=round((datalength*.40))
    else:
        traininglength=math.floor((datalength*.60))
        testlength=round((datalength*.40))
    trainingspeaker=speakers[0:traininglength-1]
    testspeakers=speakers[traininglength:datalength-1]'''
    kf = KFold(n_splits=2,shuffle=True)
    best_model=[None,None,None]
    accuracy=0
    trueactual=[]
    totalTrue=0
    totalTest=0
    truepredicted=[]
    for train_index, test_index in kf.split(speakers):
        actual=[]
        predicted=[]
        trainingspeakers=[]
        testspeakers=[]
        true=0
        test=0
        #print("Finished Training the model for:"+word+" Speakers are:")
        for train in train_index.tolist():
            trainingspeakers.append(speakers[train])
        for test in test_index.tolist():
            testspeakers.append(speakers[test])
        #print("Finished Training the model for:"+word+" Speakers are:")
        #print(trainingspeakers)
        #print("Testing Data are:")
        #print(testspeakers)
        #totalTest=totalTest+len(testspeakers)
        trainingdata=data[data["gesture"]!=word]
        for speaker in trainingspeakers:#
            trainingdata=trainingdata.append(dataperword[dataperword["speaker"]==speaker])
        training=train_all(trainingdata,x)
        for testspeaker in testspeakers:
            #print("Recognition for "+word+" speaker is: "+testspeaker)
            guessword=recognize(training,dataperword[dataperword["speaker"]==testspeaker],x)
            actual.append(word)
            predicted.append(guessword)
            test=test+1
            totalTest=totalTest+1
            if(guessword==word):
                true=true+1
                totalTrue=totalTrue+1
        if accuracy<=(true/len(testspeakers)):
            accuracy=true/len(testspeakers)
            #print(best_model)
            trueactual=actual
            truepredicted=predicted
    #print(best_model)
    best_model=list([word,x,totalTrue,totalTest,totalTrue/totalTest])
    print(str(totalTrue/totalTest))
    print("Recognized "+word+":"+ str(totalTrue)+" out of "+str(totalTest))
    dataResult=pd.DataFrame(np.array(best_model).reshape(1,5), columns = ["gesture","numHiddenStates","Total True","Total Test","Accuracy"])
    dataResult["Actual"]=0
    dataResult["Actual"]=dataResult["Actual"].astype(list)
    dataResult.at[0,"Actual"]=trueactual
    dataResult["Predicted"]=0
    dataResult["Predicted"]=dataResult["Predicted"].astype(list)
    dataResult.at[0,"Predicted"]=truepredicted
    '''dataResults=dataResults.append(dataResult)
    dataResults.to_csv("resultssample.csv",index=False)'''
    #print(dataResult)
    return dataResult
if __name__=='__main__':
    data=pd.read_csv("datacsv/datasetcomplete.csv")
    training={}
    words=data["gesture"].unique().tolist()
    totalTrue=0
    totalTest=0
    best_accuracy=0
    best_state=0
    for n_components in range(15,26):
        dataResults=pd.DataFrame(columns=["gesture","numHiddenStates","Total True","Total Test","Accuracy","Actual","Predicted"])
        wordaccuracy=[]
        start=time.time()
        actualFinal=[]
        predictedFinal=[]
        pool = multiprocessing.Pool()
        print("Starting Training at "+str(n_components)+" states")
        wordaccuracy=(pool.map(partial(testing,n_components),words))
        combined_df = pd.concat(wordaccuracy, ignore_index=True)
        totalTrue=list(map(int,combined_df["Total True"].tolist()))
        totalTest=list(map(int,combined_df["Total Test"].tolist()))
        accuracy=sum(totalTrue)/sum(totalTest)
        print(accuracy)
        if(accuracy>best_accuracy):
            best_accuracy=accuracy
            best_state=n_components
        dataResults=dataResults.append(combined_df,sort=False)
        pool.close()
        print("Took "+str(time.time()-start)+" seconds")
        dataResults.to_csv("results/complete"+str(n_components)+".csv",index=False)
    print("Best Accuracy is at "+str(best_accuracy)+" at states=" +str(best_state))
