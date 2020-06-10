#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:57:38 2020

@author: harshit
"""
import numpy as np
import pandas as pd

import time
import statistics 
import yfinance as yf

# Importing the training set
dataset_train = pd.read_csv('apple_1min.csv')
dataset_train = dataset_train.dropna()
training_set = dataset_train.iloc[:,1:].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 DAYsteps and 1 output
X_train = []
y_train = []
for i in range(120, (training_set_scaled.shape[0])):
    X_train.append(training_set_scaled[i-120:i])
    y_train.append(training_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(Bidirectional(LSTM(75, return_sequences = True),input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(Bidirectional((LSTM(75, return_sequences = True))))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(Bidirectional((LSTM(75, return_sequences = True))))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(Bidirectional(LSTM(units = 75)))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 5))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
from keras.models import load_model
regressor=load_model('biapple1.h5')


    
def predict_1min(j):
    while True:
        start_time  = time.process_time()

        data = yf.download(  # or pdr.get_data_yahoo(...
                # tickers  or string as well
                tickers = " AAPL ",
        
                # use "period" instead of start/end
                # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                # (optional, default is '1mo')
                period = "1d",
        
                # fetch data by interval (including intraday if period < 60 days)
                # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                # (optional, default is '1d')
                interval = "1m")
        
        df = pd.read_csv('apple_1_test.csv')


        data = data.drop(['Adj Close'], axis = 1)
        
        print(data.iloc[-2,:])
        
        df = df.append(data.iloc[-2,:], ignore_index = True)
        indexs = data.index
        df.iloc[-1:,0]= indexs[-2]       
        df.to_csv('apple_1_test.csv', index = False)

           
        real_stock_price = df.iloc[:, 1:].values
        #dataset_train = dataset_train.drop(col, axis =1)
        #df = df.drop(col,  axis =1)
        dataset_total = pd.concat((dataset_train, df), axis = 0, sort = False)
        inputs = dataset_total[len(dataset_total) - len(df) - 120:]
        inputs = inputs.drop(["Datetime"], axis = 1)
        #inputs = inputs.reshape(1,-1)
        inputs = sc.transform(inputs.values)
        
        X_test = []
        for i in range(120, (inputs.shape[0])):
            X_test.append(inputs[i-120:i])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        pd.DataFrame(predicted_stock_price).to_csv("prediction_1min.csv", index=False)
        
        
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%d/%m/%y %H:%M:%S")
        print ("\nPrediction at: ",current_time)
        print(predicted_stock_price[-1:])

        
        print("\nAcutal Standard Deviation")
        print(statistics.stdev(df.iloc[-5:,3]))
        
        print("\nPredicted Standard Deviation")
        print(statistics.stdev(pd.DataFrame(predicted_stock_price).iloc[-5:,3]))        
        
        j+=1
        print("\nNext value of time "+ str(indexs[-1]))

        time.sleep(60 - (time.process_time() - start_time))

j = 0 
while True:
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if current_time == "19:00:00":
        predict_1min(j)
        break
    
    

