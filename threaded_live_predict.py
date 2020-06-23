#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:23:47 2020

@author: harshit
"""


import numpy as np
import pandas as pd

import statistics 
import yfinance as yf

import time
from datetime import datetime
import threading

import tensorflow as tf
import keras

import requests, json

import os

from keras.models import model_from_json


def notify(title, text):
    os.system("""
            osascript -e 'display notification "{}" with title "{}" sound name "default"'""".format(text, title))




with open('inventory.txt', 'r') as f:
    inventory = f.readlines()

if len(inventory) >=1 :
    inventory  = list(np.float_(inventory))
    inventory.pop(0)
    

with open('investment.txt', 'r') as f:
    investment = f.read()

investment  = (np.float_(investment))

base_url = "https://paper-api.alpaca.markets"
acnt_url = "{}/v2/account".format(base_url) 
orders_url = "{}/v2/orders".format(base_url)




api_key = "PKB0KCNRWP8ZWSFKQW08"
secret_key = "0S6OpkNAXdDd7YbEB18Z3KJFMr0ZYUdoSkdlWisT"
Headers = {'APCA-API-KEY-ID' : api_key, 'APCA-API-SECRET-KEY' :secret_key }




def get_account():
        r = requests.get(acnt_url, headers = Headers)
        return json.loads(r.content)
 
r = get_account()
print(r)


def create_order(symbol, qty, side, typ, time_in_force):
    data = {  "symbol": symbol,
  "qty": qty,
  "type": typ,
  "side": side,
  "time_in_force": time_in_force,
  }
    
    r = requests.post(orders_url, json = data, headers = Headers)

    return json.loads(r.content), r.elapsed.total_seconds()

def get_position(sym):
        position_url = "{}/v2/positions/{}".format(base_url,sym)
        r = requests.get(position_url, headers = Headers)
        return json.loads(r.content), r.elapsed.total_seconds()
    
df = pd.read_csv('apple_1_test.csv')
sd = statistics.stdev(df.iloc[:,3])

avg_price = 0 
max_price = 0 
trend5 = 0 


if len(inventory)>=1:
    p, time2 = get_position("AAPL")
    avg_price =  float(p["avg_entry_price"])  + sd
    max_price = float(p["avg_entry_price"])


# Importing the training set
dataset_train1 = pd.read_csv('apple_1min.csv')
dataset_train1 = dataset_train1.dropna()
training_set1 = dataset_train1.iloc[:,1:].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc1 = MinMaxScaler(feature_range = (0, 1))
training_set1_scaled1 = sc1.fit_transform(training_set1)


# Creating a data structure with 60 DAYsteps and 1 output
X_train1 = []
y_train1 = []
for i in range(120, (training_set1_scaled1.shape[0])):
    X_train1.append(training_set1_scaled1[i-120:i])
    y_train1.append(training_set1_scaled1[i])
X_train1, y_train1 = np.array(X_train1), np.array(y_train1)

# Reshaping
X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 5))

config = tf.ConfigProto(
    device_count={'cpu': 0},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.Session(config=config)

keras.backend.set_session(session)

init = tf.global_variables_initializer()
session.run(init)

# Part 2 - Building the RNN

# load json and create model
json_file = open('regressor1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor1 = model_from_json(loaded_model_json)
# load weights into new model
regressor1.load_weights("biapple1.h5")
 

# Compiling the RNN
regressor1.compile(optimizer = 'adam', loss = 'mean_squared_error')










# Importing the training set
dataset_train5 = pd.read_csv('apple_5min.csv')
dataset_train5 = dataset_train5.dropna()
training_set5 = dataset_train5.iloc[:,1:].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc5 = MinMaxScaler(feature_range = (0, 1))
training_set_scaled5 = sc5.fit_transform(training_set5)


# Creating a data structure with 60 DAYsteps and 1 output
X_train5 = []
y_train5 = []
for i in range(120, (training_set_scaled5.shape[0])):
    X_train5.append(training_set_scaled5[i-120:i])
    y_train5.append(training_set_scaled5[i])
X_train5, y_train5 = np.array(X_train5), np.array(y_train5)

# Reshaping
X_train5 = np.reshape(X_train5, (X_train5.shape[0], X_train5.shape[1], 5))


config5 = tf.ConfigProto(
    device_count={'cpu': 0},
    intra_op_parallelism_threads=1,
)
session5 = tf.Session(config=config5)

keras.backend.set_session(session5)

init5 = tf.global_variables_initializer()
session5.run(init5)
# Part 2 - Building the RNN


json_file = open('regressor5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor5 = model_from_json(loaded_model_json)
# load weights into new model
regressor5.load_weights("apple5.h5")
regressor5.compile(optimizer = 'adam', loss = 'mean_squared_error')




def predict_1min(j,flag, trend1):
    while True:
        global stop_threads1 
        if stop_threads1: 
            print("thread 1 stopped")
            break
        else:
    
            print("\n*****1 MIN*****\n")
            j= 0 
            trend1 = 0 
            flag = 0
            if j == 4 and flag <=1:
                trend1 = 0
                j = 0
                print("\n#####reset#####\n")
            if j == 5 :
                trend1 = 0
                j = 0
                print("\n#####reset#####\n")
                
            data1 = yf.download(  # or pdr.get_data_yahoo(...
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
            
            df1 = pd.read_csv('apple_1_test.csv')
    
    
                    
            # dropping duplicate values 
            
            data1 = data1.drop(['Adj Close'], axis = 1)
            
            print(data1.iloc[-2,:])
            
            df1 = df1.append(data1.iloc[-2,:], ignore_index = True)
            indexs = data1.index
            df1.iloc[-1:,0]= indexs[-2]       
            df1.drop_duplicates(keep='first',inplace=True)
    
            df1.to_csv('apple_1_test.csv', index = False)
    
            
            dataset_train1 = pd.read_csv('apple_1min.csv')
            dataset_train1 = dataset_train1.dropna()
            dataset_total1 = pd.concat((dataset_train1, df1), axis = 0, sort = False)
            inputs1 = dataset_total1[len(dataset_total1) - len(df1) - 120:]
            inputs1 = inputs1.drop(["Datetime"], axis = 1)
            inputs1 = sc1.transform(inputs1.values)
            
            X_test1 = []
            for i in range(120, (inputs1.shape[0])):
                X_test1.append(inputs1[i-120:i])
    
            X_test1 = np.array(X_test1)
            X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 5))
            with session.as_default():
                with session.graph.as_default():
                    predicted_stock_price1 = regressor1.predict(X_test1)
                    
            predicted_stock_price1 = sc1.inverse_transform(predicted_stock_price1)
    
            pd.DataFrame(predicted_stock_price1).to_csv("prediction_1min.csv", index=False)
            
            trend1 = trend1 +  predicted_stock_price1[-2,0] - predicted_stock_price1[-1,0]
            text_file = open("trend1.txt", "w")
            text_file.write(str(trend1))
            text_file.close()   
            
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%d/%m/%y %H:%M:%S")
            print ("\n1 Minute Prediction at: ",current_time)
            print(predicted_stock_price1[-1:,:])
    
    
            j+=1
            flag = flag + 0.25
    
    
            print("Trend: ", trend1)
            print("\nNext value of time "+ str(indexs[-1]))
            
            time.sleep(59.5)
            
def predict_5min(j,investment):
    while True:
        global stop_threads2
        if stop_threads2: 
            print("thread 2 stopped")
            break
        else:
            start_time = time.process_time()
            trend5 = 0
            trend1 = 0
            time1 = 0
            time2 = 0
            print("\n*****5 MIN*****\n")
    
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
                    interval = "5m")
            df = pd.read_csv('apple_5_test.csv')
    
            data = data.drop(['Adj Close'], axis = 1)
            
            print(data.iloc[-2,:])
            
            
            df = df.append(data.iloc[-2,:], ignore_index = True)
            indexs = data.index
            df.iloc[-1:,0]= indexs[-2]
            
            df.drop_duplicates(keep='first',inplace=True)
    
            df.to_csv('apple_5_test.csv', index = False)
            
                    
            dataset_train = pd.read_csv('apple_5min.csv')
            dataset_train = dataset_train.dropna()
            
            dataset_total = pd.concat((dataset_train, df), axis = 0, sort = False)
            inputs = dataset_total[len(dataset_total) - len(df) - 120:]
            inputs = inputs.drop(["Datetime"], axis = 1)
            inputs = inputs.values
            #inputs = inputs.reshape(1,-1)
            inputs = sc5.transform(inputs)
            X_test = []
            for i in range(120, (inputs.shape[0])):
                X_test.append(inputs[i-120:i])
                
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
            
            with session5.as_default():
                with session5.graph.as_default():
                    predicted_stock_price = regressor5.predict(X_test)
            predicted_stock_price = sc5.inverse_transform(predicted_stock_price)
        
            
            trend5 = predicted_stock_price[-2,0] - predicted_stock_price[-1,0]
            
            #acnt = get_account()
            #cash = float(acnt["cash"])
            
            '''
            
            values_1 = pd.read_csv("apple_1_test.csv").values
            predict1 = pd.read_csv("prediction_1min.csv").values
            
            values_2 = values_1[-5:-1,3]
            for i in range(3):
                trend1 = trend1  + values_2[i+1] - values_2[i]   
                
            trend1 = trend1 + values_1[-1,3] - predict1[-1,3]
            
            trend1 = np.subtract(values_1[-5:,3],values)
    '''
            f = open("trend1.txt", "r")
            trend1 = float(f.read())
            
            if (trend5 <= 0 and trend1 <=0):
    
                respone, time1 = create_order("AAPL", 1, "buy", "market", "day")
                r, time2 = get_position("AAPL") 
                
                f = open("investment.txt", "r")
                investment = np.float(f.read())
                

                
                with open('inventory.txt', 'r') as f:
                    inventory = f.readlines()
                
                if len(inventory) >=1 :
                    inventory  = list(np.float_(inventory))
                    inventory.pop(0)
                    
                print("I Investemnt :", investment)
                print(float(r["current_price"]))
                investment = investment + float(r["current_price"])
                text_file = open("investment.txt", "w")
                text_file.write(str(investment))
                text_file.close()   
                
                inventory.append(float(r["current_price"]))
                print("Total Investment:", investment)
           
                pd.DataFrame(inventory).to_csv("inventory.txt", index = False)
           
                print (respone )
                notify("Buy",float(r["current_price"]))
    
                print("\nBuying Stock")
    
    
            if((trend5 < 0 and trend1 > 0 ) or (trend5 > 0 and trend1 < 0) ):
                print("\nHolding Stock")
                
            text_file = open("trend5.txt", "w")
            text_file.write(str(trend5))
            text_file.close()   
                
            pd.DataFrame(predicted_stock_price).to_csv("prediction_5min.csv", index=False)
    
            now = datetime.now()
            current_time = now.strftime("%d/%m/%y %H:%M:%S")
            
            print ("\n5 Minute Prediction at: ",current_time)
            print(predicted_stock_price[-1:])
            
            '''
            print("\nAcutal Standard Deviation")
            print()
            
            print("\nPredicted Standard Deviation")
            print(statistics.stdev(pd.DataFrame(predicted_stock_price).iloc[-5:,3]))
            
            '''
            
            print("\nNext value of time"+ str(indexs[-1]))
            
            print("\nTrend1: ", trend1 ," Trend5 : ", trend5)
            
            time.sleep(300 - time1 - (time.process_time() - start_time) - time2)
           


def selling(max_price):
    while True:
        global stop_threads3 
        if stop_threads3: 
            print("thread 3 stopped")
            break
        else:
            
            time.sleep(30)
        
            df = pd.read_csv('apple_1_test.csv')
            sd = statistics.stdev(df.iloc[:,1])
            
            with open('inventory.txt', 'r') as f:
                 inventory = f.readlines()
                 
            inventory  = list(np.float_(inventory))
            inventory.pop(0)         
               
           
           
            if len(inventory) >=1:
                
                
                r1, time2 = get_position("AAPL")
                
                avg_price =  float(r1["avg_entry_price"])
                avg_price = avg_price + sd  
                if( (float(r1["current_price"])< avg_price) ):
                    
                    respone,time1 = create_order("AAPL", len(inventory), "sell", "market", "day")
                    for i in range(len(inventory)):
                        inventory.pop(0)
                    investment = 0
                    inventory = 0

                    text_file = open("investment.txt", "w")
                    text_file.write(str(investment))
                    text_file.close() 
                    text_file = open("inventory.txt", "w")
                    text_file.write(str(inventory))
                    text_file.close()                     
                    print (respone)
                    
                    notify("Sell", str(r1["current_price"]))

                    print("\n************Selling Stock avg price************\n") 
            
                    inventory = []
            
            if len(inventory) >=1:
                r1, time2 = get_position("AAPL")

                
                f = open("investment.txt", "r")
                investment = np.float(f.read())
                
                value = 0   
                value = investment - (0.05 * investment)
                
                if( (value > float(r1["market_value"]))):    
                
                    respone,time1 = create_order("AAPL", len(inventory), "sell", "market", "day")
                    for i in range(len(inventory)):
                        inventory.pop(0)
                    inventory = 0
                    investment = 0
                    text_file = open("investment.txt", "w")
                    text_file.write(str(investment))
                    text_file.close() 
                    text_file = open("inventory.txt", "w")
                    text_file.write(str(inventory))
                    text_file.close()                         
                    print (respone)
                    
                    notify("Sell", str(r1["current_price"]))
                    print("\n************Selling Stock investment************\n") 
                    inventory = []
                      
            if len(inventory) >=1:
                
                
                if max_price < float(r1["current_price"]):
                    max_price = float(r1["current_price"])
                    
                r1, time2 = get_position("AAPL")

                
                f = open("trend1.txt", "r")
                trend1 = float(f.read())
                
                f = open("trend5.txt", "r")
                trend5 = float(f.read())

                if (( (max_price + sd) < float(r1["current_price"]) >= max_price) and  (trend5 > 0 and trend1> 0)):
                    respone,time1 = create_order("AAPL", len(inventory), "sell", "market", "day")
                    for i in range(len(inventory)):
                        inventory.pop(0)   
                    inventory = 0

                    investment = 0
                    text_file = open("investment.txt", "w")
                    text_file.write(str(investment))
                    text_file.close()   
                    text_file = open("inventory.txt", "w")
                    text_file.write(str(inventory))
                    text_file.close()  
                    print (respone)
                    max_price = 0
                    notify("Sell", str(r1["current_price"]))

                    print("\n************Selling Stock max price************\n")  
                    inventory = []

        
stop_threads1 = True
stop_threads2 = True 
stop_threads3 = True

flag  = 0 

      
   
while True:
    now = datetime.now()
    current_time = now.strftime("%M:%S")
    if current_time == "31:02":
        stop_threads1 = False
        t1 = threading.Thread(target=predict_1min, args=(0, flag, 0))     
        t1.start()
        stop_threads3 = False
        t3 = threading.Thread(target=selling, args=(max_price,))     
        t3.start()
        breakq
    
while True:
    now = datetime.now()
    current_time = now.strftime("%M:%S")
    if current_time == "34:45":
        stop_threads2 = False
        t2 = threading.Thread(target=predict_5min, args=(0,investment))  
        t2.start()
        break 

"""# wait until thread 1 is completely executed 
t1.join() 
# wait until thread 2 is completely executed 
t2.join() """