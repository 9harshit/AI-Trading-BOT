#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:39:51 2020

@author: harshit
"""


import statistics 
import pandas as pd
# creating a simple data - set 
dataset_train = pd.read_csv('apple_5min.csv')

# Getting the real stock price of 2017
dataset_test = pd.read_csv('apple_5_test.csv')
real_stock_price = dataset_test.iloc[:, 1:].values
#dataset_train = dataset_train.drop(col, axis =1)
#dataset_test = dataset_test.drop(col,  axis =1)
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0, sort = False)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]# Prints standard deviation 
inputs = inputs.drop(['date'], axis =1)
# xbar is set to default value of 1 
inputs = inputs.values
col = inputs[30:,0]
print(statistics.stdev(col))