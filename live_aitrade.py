#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:44:02 2020

@author: harshit
"""



import pandas as pd
import requests,json
import time
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from collections import deque
import yfinance as yf


action_space = 3
memory = deque(maxlen = 2000) #EXPERIENCE REPLACY
inventory = [] #STOCKS WE OWN

with open('inventory_5min.txt', 'r') as f:
    inventory = f.readlines()

if len(inventory) >1 :
    inventory  = list(np.float_(inventory))
    inventory.pop(0)
    
timestep = 12
window_size = timestep + 1


gamma = 0.90
epsilon = 0.05

base_url = "https://paper-api.alpaca.markets"
acnt_url = "{}/v2/account".format(base_url) 
orders_url = "{}/v2/orders".format(base_url)

api_key = "PKSOHEJ8K0OSXT1G09MV"
secret_key = "/WoCAmSF5ggfSYTZxru2k3QyxenQYq4k5HJBCDGc"
Headers = {'APCA-API-KEY-ID' : api_key, 'APCA-API-SECRET-KEY' :secret_key }


def get_account():
        r = requests.get(acnt_url, headers = Headers)
        return json.loads(r.content)
 
r = get_account()
print(r)


dataset_train = pd.read_csv('apple_5min.csv')
dataset_train = dataset_train.dropna()
training_set = dataset_train.iloc[:,1:].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#MODEL WILL RETURN WILL ACTION TO PERFROM AND INPUT IS State/prices
        
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True, input_shape = (timestep, 5)))
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
regressor.add(Dense(units = action_space, activation = "linear"))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')#OUTPUT AS SELL, BUY, HOLD

        
from keras.models import load_model
regressor=load_model('trader.h5')



      

#TRADE FUNCTION TAKES STATE AS AN INPUT AND OUTPUT IS ACTION TO PERFROM ACTION IN PARTICULAR STATE
def trade(state):
    print(state)
    action = regressor.predict(state)
    print(action)
    #WILL RETURN ACTION WITH HIGHEST PROBABILITY
    return np.argmax(action[0]) #0 because of output shape

def stock_price_format(n):
    if n < 0:
        return "-${:.2f}".format(abs(n))

    else:
        return "${:.2f}".format(abs(n))
    
    
def state_creator(data, timestep):
    
    state = data[-(timestep+1):,:]


    return np.array(state)



def reshaping_state(data, rnn_tmstp):
    data = sc.transform(data)

    state_reshaped = []
    
    for i in range(rnn_tmstp, (data.shape[0])):
        state_reshaped.append(data[i-rnn_tmstp+1:])
        
    state_reshaped = np.array(state_reshaped)
    state_reshaped = np.reshape(state_reshaped, (state_reshaped.shape[0], state_reshaped.shape[1], 5))
    print(state_reshaped)
    return state_reshaped

 

def create_order(symbol, qty, side, typ, time_in_force):
    data = {  "symbol": symbol,
  "qty": qty,
  "type": typ,
  "side": side,
  "time_in_force": time_in_force,
  }
    
    r = requests.post(orders_url, json = data, headers = Headers)

    return json.loads(r.content)


def live():
    total_profit = 0

    while True:
        start_time = time.process_time()
        data = yf.download(  # or pdr.get_data_yahoo(...
            # tickers  or string as well
            tickers = "AAPL",
        
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
        
        df.to_csv('apple_5_test.csv', index = False)
        
        dataset_total = pd.concat((dataset_train, df), axis = 0, sort = False)
        inputs = dataset_total[len(dataset_total) - len(df) - 120:]
        inputs = inputs.drop(["Datetime"], axis = 1)
        inputs = inputs.values
        #inputs = inputs.reshape(1,-1)
        
        state = state_creator(inputs, timestep)
        
        state_reshaped = reshaping_state(state, timestep)
        
        
        action = trade(state_reshaped) 
        acnt = get_account()
        cash = float(acnt["cash"])


        if action == 1 and cash >= inputs[-1,3] : 
    
            respone = create_order("AAPL", 1, "buy", "market", "day")
            
            inventory.append((inputs[-1,3]))
            
            print("Bought at : ", stock_price_format(inputs[-1,3]))
            pd.DataFrame(inventory).to_csv("inventory_5min.txt", index = False)

            print (respone )
            
        if action == 2 and len(inventory) >= 1:
            
            respone = create_order("AAPL", 1, "sell", "market", "day")
            buy_price = inventory.pop(0)
            pd.DataFrame(inventory).to_csv("inventory_5min.txt", index = False)

            print("Bought at:", stock_price_format(buy_price))
            print("Sold at:", (inputs[-1,3]))
            
            total_profit += inputs[-1,3] - buy_price
            print(total_profit)
           # print("Apple Stock Bought at " + stock_price_format(buy_price),"Sold at " + stock_price_format((training_set[-1,3])),"Profit " + stock_price_format((training_set[-1,3] - buy_price)))
    
            print (respone)
            
        if action == 0 :
            print("Holding Apple Stock")
        
        print("action: ",action)

    
        print("\nNext value of time"+ str(indexs[-1]))
        
        
        time.sleep(300 - (time.process_time() - start_time))


while True:
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if current_time == "20:27:03":
        live()
        break
    
