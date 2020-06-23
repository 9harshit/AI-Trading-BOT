#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:00:04 2020

@author: harshit
"""


import matplotlib.pyplot as plt
import pandas as pd


real_stock_price = pd.read_csv('apple_5_test.csv')
predicted_stock_price = pd.read_csv('prediction_5min.csv').values


plt.plot(real_stock_price.iloc[1:,4], color = 'red', marker = 'o', label = 'Real  apple Stock High Price')
plt.plot(predicted_stock_price[:,3], color = 'blue', marker = 'o',label = 'Predicted  apple High Stock Price')
plt.title('M ahindra  Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel(' apple  Stock Price')
plt.legend()
plt.show()



real_stock_price = pd.read_csv('apple_1_test.csv')
predicted_stock_price = pd.read_csv('prediction_1min.csv').values


plt.plot(real_stock_price.iloc[1:,4], color = 'red', marker = 'o', label = 'Real  apple Stock High Price')
plt.plot(predicted_stock_price[:,3], color = 'blue', marker = 'o',label = 'Predicted  apple High Stock Price')
plt.title('M ahindra  Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel(' apple  Stock Price')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import pandas as pd


real_stock_price = pd.read_csv('coca_5_test.csv')
predicted_stock_price = pd.read_csv('prediction_5min_coca.csv').values


plt.plot(real_stock_price.iloc[1:,3], color = 'red', marker = 'o', label = 'Real  apple Stock High Price')
plt.plot(predicted_stock_price[:,3], color = 'blue', marker = 'o',label = 'Predicted  apple High Stock Price')
plt.title('M ahindra  Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel(' apple  Stock Price')
plt.legend()
plt.show()



real_stock_price = pd.read_csv('coca_1_test.csv')
predicted_stock_price = pd.read_csv('prediction_1min_coca.csv').values


plt.plot(real_stock_price.iloc[1:,3], color = 'red', marker = 'o', label = 'Real  apple Stock High Price')
plt.plot(predicted_stock_price[:,3], color = 'blue', marker = 'o',label = 'Predicted  apple High Stock Price')
plt.title('M ahindra  Stock Price Prediction')
plt.xlabel('DAY')
plt.ylabel(' apple  Stock Price')
plt.legend()
plt.show()