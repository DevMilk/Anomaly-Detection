# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:29:08 2020

@author: Ugur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
 
os.chdir(os.getcwd())
figDir = "Figures/"     


df = pd.read_json('syntetic-nd.json', lines = True) 
df = df.set_index("timestamp")
#%%
from sklearn import preprocessing

def split(): print("\n____________________________________________________________________________________\n")

#Tüm featureler için korelasyon matrisi
def plotCorrelationMatrix(df, graphWidth,save=False):  
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for xd', fontsize=40)
    plt.show()
#Boxplot, gruplama ve korelasyonların hepsini analiz eden all-in-one fonksiyon
    if(save):
        plt.savefig(figDir+'CorrelationMatrix.png')
def intro(df,graph=True,splitPlots=True,EraseNullColumns=False,printCorrelations=True,corrThreshold=0.5,save=False):
    
    dataframe=df.copy()
    
    if(EraseNullColumns==True):  dataframe.dropna(axis=1,inplace=True)

    split()
    print(df)
    split()
    print(dataframe.head(5))
    split()
    
    print(dataframe.info())
    split()
    
    print(dataframe.describe())
    split()
    
#-------------------------------BOXPLOTFEATURES-----------------------------      
    
    
    if(graph):

        if(splitPlots==True):
            print("                         ___BOXPLOTFETURES")

            for column in dataframe.columns:
                if(dataframe[column].dtype==np.int or dataframe[column].dtype==np.float):
                    plt.figure()
                    dataframe.boxplot([column])
                    if(save):
                        plt.savefig(figDir+'{}.png'.format(column))
                    
        else:
            dataframe.boxplot()
            
    #If unique values of columns is under 10, print unique values with considered column


#-------------------------------GROUPBY-----------------------------        

    print("                         _____GROUPBY____")

    for column in dataframe.columns:    
        unique_values=dataframe[column].unique()
        if(unique_values.size<=10):
            print(column,": ",unique_values)
            print("\nGrouped By: ",column,"\n\n",dataframe.groupby(column).mean())
            split()
            print("\n")
            
        
#-------------------------------CORRELATIONS-----------------------------        
    if(printCorrelations==True):
        print("                         ____CORRELATIONS____")
        corrByValues= dataframe.corr().copy()
        flag = False
        corr_matrix=abs(corrByValues>=corrThreshold)
        columns= corr_matrix.columns
        for i in range(columns.size):
            for j in range(i,columns.size):
                iIndex=columns[i]
                jIndex=columns[j] 
                if (i!=j and corr_matrix[iIndex][jIndex]==True and (len(df[iIndex].unique())!=1 and len(df[jIndex].unique())!=1 )):
                    sign = "Positive"
                    if(corrByValues[iIndex][jIndex]<0): sign="Negative"
                    split()
                    flag = True
                    print(iIndex.upper(), " has a " ,sign," correlation with ",jIndex.upper(),": {} \n".format(corrByValues[iIndex][jIndex]))
        
        plt.show()
        plotCorrelationMatrix(df,30)       
        
        split()
        if(not flag):
            print("No Correlation Found") 
    return dataframe

#KDE dağılımı ile featureları plotlar
def plotCols(df,time,save=False):
    
    for col in df.columns:
        if(df[col].dtype==np.int or df[col].dtype==np.float):
            if(len(df[col].unique())>1):
                fig = df.plot(x=time,y=col,kind="kde", title = "{}-{} KDE".format(time,col))   
                if(save):
                    fig.get_figure().savefig(figDir+"{}-kde.png".format(time+"-"+col))
                plt.show()
            plt.plot(df[time],df[col]) 
            plt.title("{}-{}".format(time,col))
            plt.show()
            if(save):
                plt.savefig(figDir+'{}.png'.format(time+"-"+col))
        
#Verilen feature'ları scatter ile Y'ye göre karşılaştırır.        
def XCorrWithY(df, X, Y):
    for col in  X:
        print(col,"-",Y)
        plt.scatter(df[col],df[Y]) 
        plt.title("{}-{}".format(col,Y))
        plt.show()    
#Dataframeyi normalize eder. (Preprocessing)        
def normalizedf(df,offset=0):
    min_max_scaler = preprocessing.MinMaxScaler() 
    new = df.copy()
    cols = df.columns[offset:]
    new[cols] = (min_max_scaler.fit_transform(new[cols]) )  
    return new    
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset.iloc[i:(i+look_back), :]
		dataX.append(a.to_numpy())
		dataY.append(dataset.iloc[i + look_back, -1]) 
	return np.array(dataX), np.array(dataY)  


#%% Preprocessing

df = normalizedf(df)

X_f, Y_f = create_dataset(df.drop(columns=["Feature1"]),17)
#%% Visualize and Analyze dataframe
intro(df)
df.reset_index().plot(x="timestamp",y="Feature1",kind="scatter")
df.reset_index().plot(x="timestamp",y="Feature2",kind="scatter")

#Bir senörün anomlisi, diğer sensörden bağımsız. Anomalileri arasında ilişkileri yok.
#2 Sensör arasında Korelasyonun 0.54 değerinde olması, frekanslarının aynı olmasından kaynaklanıyor olmalı
#%% Sklearn Anomaly Detection Libraries
from sklearn.svm import OneClassSVM #Bu data üzerinde Başarılı Değil
from sklearn.ensemble import IsolationForest  #Bu data üzerinde başarılı değil
from sklearn.neighbors import LocalOutlierFactor 
modelsNotFitted = [ LocalOutlierFactor(leaf_size=100 , novelty=False)] 
X = X_f.reshape(X_f.shape[:-1]) 
#Anomali olup olmadığına bak
for model in modelsNotFitted: 
    Y = model.fit_predict(X)
    plt.plot(Y)
    plt.plot(Y_f)
    plt.show()    
    
#LocalOutlierFactor çok başarılı!    
    #%%
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM, Dropout, LeakyReLU  

from keras.optimizers import Adam, SGD, Adamax,RMSprop  
import tensorflow as tf 
from keras.callbacks import EarlyStopping, ModelCheckpoint

testX, testY = X_f[-2000:],Y_f[-2000:]
trainX, trainY = X_f[:-2000], Y_f[:-2000]  
m = Sequential()
m.add(LSTM(50, input_dim=(1), return_sequences=True))
m.add(LSTM(25, return_sequences=False )) 
m.add(Dense(1))
m.compile(loss='mse', optimizer='adam')
m.fit(trainX, trainY,use_multiprocessing = True, validation_data=(testX,testY),workers = 2,epochs = 200, batch_size= 1000, callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min'),
                       ModelCheckpoint("model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=1)])
#%%
m = load_model("model.h5")
plt.plot(Y_f,label="Real" )
plt.show()
plt.plot(m.predict(X_f),label="predict")
plt.show()
 

#%%
import math
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose

autocorrelation_plot(X)
decompose = seasonal_decompose(X,model="additive",period=5000 ).plot() #Yıllık olarak dalganın aayrıştıırlması 
