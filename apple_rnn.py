#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:42:36 2020

@author: harshit
"""


# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('apple_5min.csv')
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
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 5))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 90, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('apple_5_test.csv')
real_stock_price = dataset_test.iloc[:, 1:].values
#dataset_train = dataset_train.drop(col, axis =1)
#dataset_test = dataset_test.drop(col,  axis =1)
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0, sort = False)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:]
inputs = inputs.drop(["Datetime"], axis = 1)
inputs = inputs.values
#inputs = inputs.reshape(1,-1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120,  (inputs.shape[0])):
    X_test.append(inputs[i-120:i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#predicted_stock_price.to_csv('open-low')
# Visualising the results 
'''
plt.plot(real_stock_price[1:,0], color = 'red', marker = 'o', label = 'Real  apple tock Open Price')
plt.plot(predicted_stock_price[:,0], color = 'blue', marker = 'o',label = 'Predicted  apple Open Stock Price')
plt.title('applen Paint Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel('applen Paint Stock Price')
plt.legend()
plt.show()


plt.plot(real_stock_price[1:,1], color = 'red', marker = 'o', label = 'Real  apple Stock High Price')
plt.plot(predicted_stock_price[:,1], color = 'blue', marker = 'o',label = 'Predicted  apple High Stock Price')
plt.title('M ahindra  Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel(' apple  Stock Price')
plt.legend()
plt.show()


plt.plot(real_stock_price[1:,2], color = 'red',  marker = 'o',label = 'Real  apple Stock Low  Price')
plt.plot(predicted_stock_price[:,2], color = 'blue', marker = 'o',label = 'Predicted apple Stock Low Price')
plt.title(' apple Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel(' apple Stock Price')
plt.legend()
plt.show()

plt.plot(real_stock_price[1:,3], color = 'red', marker = 'o',label = 'Real  apple Close Price')
plt.plot(predicted_stock_price[:,3], color = 'blue', marker = 'o',label = 'Predicted  apple Stock Close Price')
plt.title(' apple  Stock Price Prediction') 
plt.xlabel('Date')
plt.ylabel(' apple Stock Price')

#plt.xticks(date.index,date['Date'].values)
plt.legend()
plt.show()

pd.DataFrame(predicted_stock_price).to_csv("predicted_values_apple5.csv")
#regressor.save("apple5.h5")
'''
from keras.models import load_model
regressor=load_model('apple1.h5')