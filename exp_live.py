#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:26:04 2020

@author: harshit
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:23:47 2020

@author: harshit
"""
import os

def notify(title, text):
    os.system("""
            osascript -e 'display notification "{}" with title "{}" sound name "default"'""".format(text, title))




import numpy as np
import pandas as pd

import statistics 
import yfinance as yf

import time
from datetime import datetime
import threading

import requests, json

with open('inventoryc.txt', 'r') as f:
    inventory = f.readlines()

if len(inventory) >0 :
    inventory  = list(np.float_(inventory))
    inventory.pop(0)
    
trend5 = 0 


stock = "AAPL"

base_url = "https://paper-api.alpaca.markets"
acnt_url = "{}/v2/account".format(base_url) 
orders_url = "{}/v2/orders".format(base_url)

api_key = "PKDE6Q20OKDDJYRXOFHL"
secret_key = "KpQ2pSaNeVCZzMzYRHO3hqNJdWAiCnA1eV75R9cD"
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

    return json.loads(r.content)


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
import tensorflow as tf
import keras



config = tf.ConfigProto(
    device_count={'cpu': 0},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.Session(config=config)

keras.backend.set_session(session)

init = tf.global_variables_initializer()
session.run(init)


from keras.models import model_from_json
 
 
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
X_train = np.reshape(X_train5, (X_train5.shape[0], X_train5.shape[1], 5))

config5 = tf.ConfigProto(
    device_count={'cpu': 0},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session5 = tf.Session(config=config5)

keras.backend.set_session(session5)

init5 = tf.global_variables_initializer()
session5.run(init5)


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
            break
        else:
            print("\n*****1 MIN*****\n")
            if j == 4 and flag <=1:
                trend1 = 0
                j = 0
                print("\nreset\n")
            if j == 5 :
                trend1 = 0
                j = 0
                print("\nreset\n")
    
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
            
            df1 = pd.read_csv('apple_1_test2.csv')
    
                    
            # dropping duplicate values 
            df1.drop_duplicates(keep='first',inplace=True)
            
            data1 = data1.drop(['Adj Close'], axis = 1)
            
            print(data1.iloc[-2,:])
            
            df1 = df1.append(data1.iloc[-2,:], ignore_index = True)
            indexs = data1.index
            df1.iloc[-1:,0]= indexs[-2]       
            df1.to_csv('apple_1_test2.csv', index = False)
    
            
            dataset_train1 = pd.read_csv('apple_1min2.csv')
            dataset_train1 = dataset_train1.dropna()
            dataset_total1 = pd.concat((dataset_train1, df1), axis = 0, sort = False)
            inputs = dataset_total1[len(dataset_total1) - len(df1) - 120:]
            inputs1 = inputs.drop(["Datetime"], axis = 1)
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
    
            pd.DataFrame(predicted_stock_price1).to_csv("prediction_1min_apple2.csv", index=False)
            
            trend1 = trend1 + predicted_stock_price1[-2,0] - predicted_stock_price1[-1,0]
            text_file = open("trend12.txt", "w")
            text_file.write(str(trend1))
            text_file.close()   
            
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%d/%m/%y %H:%M:%S")
            print ("\nPrediction at: ",current_time)
            print(predicted_stock_price1[-1:,:])
    
    
            j+=1
            flag = flag + 0.25
    
    
            print("Trend: ", trend1)
            print("\nNext value of time "+ str(indexs[-1]))
            time.sleep(59)

        
def predict_5min(j):
    while True:
        global stop_threads5 
        if stop_threads5: 
            break
        else:
            trend5 = 0
            trend1 = 0
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
            
            df = pd.read_csv('apple_5_test2.csv')
    
            data = data.drop(['Adj Close'], axis = 1)
            
            print(data.iloc[-2,:])
            
            
            df = df.append(data.iloc[-2,:], ignore_index = True)
            indexs = data.index
            df.iloc[-1:,0]= indexs[-2]
            
            df.drop_duplicates(keep='first',inplace=True)
    
            df.to_csv('apple_5_test2.csv', index = False)
            
                    
            dataset_train = pd.read_csv('apple_5min2.csv')
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
            f = open("trend12.txt", "r")
            trend1 = float(f.read())
            
            if (trend5 >= 0 and trend1 >=0  and len(inventory) >= 1):
                 
                respone = create_order("AAPL", len(inventory), "sell", "market", "day")
                for i in range(len(inventory)):
                        inventory.pop(0)  
                pd.DataFrame(inventory).to_csv("inventoryc.txt", index = False)
        
                print (respone)
                  
                print("\nSelling Stock")
                
                notify("SELL", str(df.iloc[-1,3]))
    
                
                #and cash >= df.iloc[-1,3]
            if (trend5 < 0 and trend1 < 0 ):
                  
                respone = create_order("AAPL", 1, "buy", "market", "day")
                  
                inventory.append(df.iloc[-1,3])
                
                pd.DataFrame(inventory).to_csv("inventoryc.txt", index = False)
           
                print (respone )
                
                print("\nBuying Stock")
                
                notify("BUY", str(df.iloc[-1,3]))
    
                
            if((trend5 < 0 and trend1 > 0 ) or (trend5 > 0 and trend1 < 0) ):
                print("\nHolding Stock")
                
            pd.DataFrame(predicted_stock_price).to_csv("prediction_5min_apple2.csv", index=False)
    
            now = datetime.now()
            current_time = now.strftime("%d/%m/%y %H:%M:%S")
            
            print ("\nPrediction at: ",current_time)
            print(predicted_stock_price[-1:])
            
            '''
            print("\nAcutal Standard Deviation")
            print(statistics.stdev(df.iloc[-5:,3]))
            
            print("\nPredicted Standard Deviation")
            print(statistics.stdev(pd.DataFrame(predicted_stock_price).iloc[-5:,3]))
            
            '''
            
            print("\nNext value of time"+ str(indexs[-1]))
            
            print("\nTrend1: ", trend1 ," Trend5 : ", trend5)
    
            time.sleep(297)
            
            

  
    
flag  = 0 

 
stop_threads1 = True
stop_threads5 = True


while True:
    now = datetime.now()
    current_time = now.strftime("%M:%S")
    if current_time == "51:02":
        stop_threads1 = False
        t1 = threading.Thread(target=predict_1min, args=(0, flag, 0))     
        t1.start()
        break
    
while True:
    now = datetime.now()
    current_time = now.strftime("%M:%S")
    if current_time == "54:45":
        stop_threads5 =  False
        t2 = threading.Thread(target=predict_5min, args=(0,))  
        t2.start()
        break

"""# wait until thread 1 is completely executed 
t1.join() 
# wait until thread 2 is completely executed 
t2.join() """