#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:13:56 2020

@author: harshit
"""

import datetime

dataset = data[(list(data))[1]]
line  = dataset[(list(dataset))[0]]

d = datetime.datetime.strptime((list(dataset))[0], '%Y-%m-%d %H:%M:%S')
date = datetime.date.strftime(d, "%d/%m/%y %H:%M")
print (date)
print ("Current stock price:")
print(line)
line.update({'date':date})

df = pd.read_csv('apple_5_test.csv')
df = df.append(line, ignore_index=True)
df.to_csv('apple_5_test_live.csv',index=False)
        
stock = "KO"
save = "coca"
        
import yfinance as yf
import pandas as pd

data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers  or string as well
        tickers = stock ,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "7d",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1m")

data = data.values
data = data.drop(['Adj Close'], axis = 1)
indexs = data.index

data.iloc[:-5,:].to_csv(save + '_1min.csv')
data.iloc[-5:-1,:].to_csv(save + '_1_test.csv')


data.iloc[:-5,:].to_csv(save + '_1min2.csv')
data.iloc[-5:-1,:].to_csv(save + '_1_test2.csv')


import yfinance as yf
import pandas as pd

data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers  or string as well
        tickers = stock,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "1mo",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "5m")

data = data.drop(['Adj Close'], axis = 1)

print(data.iloc[-1,:])

data.iloc[:-5,:].to_csv(save + '_5min.csv')
data.iloc[-5:-1,:].to_csv(save + '_5_test.csv')

data.iloc[:-5,:].to_csv(save + '_5min2.csv')
data.iloc[-5:-1,:].to_csv(save + '_5_test2.csv')




stock = "AAPL"
save = "apple"
        
import yfinance as yf
import pandas as pd

data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers  or string as well
        tickers = stock ,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "7d",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1m")

data = data.drop(['Adj Close'], axis = 1)
indexs = data.index

data.iloc[:-5,:].to_csv(save + '_1min.csv')
data.iloc[-5:-1,:].to_csv(save + '_1_test.csv')


data.iloc[:-5,:].to_csv(save + '_1min2.csv')
data.iloc[-5:-1,:].to_csv(save + '_1_test2.csv')

import yfinance as yf
import pandas as pd

data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers  or string as well
        tickers = stock,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "1mo",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "5m")

data = data.drop(['Adj Close'], axis = 1)

print(data.iloc[-1,:])

data.iloc[:-5,:].to_csv(save + '_5min.csv')
data.iloc[-5:-1,:].to_csv(save + '_5_test.csv')

data.iloc[:-5,:].to_csv(save + '_5min2.csv')
data.iloc[-5:-1,:].to_csv(save + '_5_test2.csv')


print(data.iloc[-2,:])
# Pass the row elements as key value pairs to append() function 
df = pd.read_csv('apple_1_test_live.csv')

df = df.append(data.iloc[-1,:], ignore_index = True)

df.iloc[-1:,0]= indexs[-1]

df.to_csv('apple_1_test_live.csv')







df = pd.read_csv('apple_5_test.csv')
df.append(data.iloc[-1,:], ignore_index = True)
df.to_csv('apple_5_test_live.csv')



data = data.drop(['Adj Close'], axis =1)
data.to_csv('apple_1_test.csv')
data['date'] = data.index.values

import pandas as pd
df = pd.read_csv('apple_1_test.csv')
df = df.append(data, ignore_index=True)
str1 = demo[2:12] + " " +demo[13:18]

import datetime
d = datetime.datetime.strptime(str1, "%Y-%m-%d %H:%M")
date = datetime.date.strftime(d, "%d/%m/%y %H:%M")
print (date)


  
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
ts = TimeSeries(key='H3JHR05VIDLIPGU0', output_format='pandas')
data1, meta_data1 = ts.get_intraday(symbol='AAPL',interval='1min', outputsize='full')
print(data1.head(-1))

data1.to_csv('new1.csv')

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
ts = TimeSeries(key='H3JHR05VIDLIPGU0', output_format='pandas')
data5, meta_data5 = ts.g et_intraday(symbol='AAPL',interval='5min', outputsize='full')
print(data5.head(-1))

data5.to_csv('new5.csv')


data1.to_csv('apple_1min.csv')
data5.to_csv('apple_5min.csv')

#Getting lastest value from the api

import pandas as pd
import requests,json
r = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&outputsize=compact&interval=1min&apikey=H3JHR05VIDLIPGU0&datatype=json")
# Print the status code of the response.
data = json.loads(r.content)
dataset = data[(list(data))[1]]
line  = dataset[(list(dataset))[0]]

import datetime
d = datetime.datetime.strptime((list(dataset))[0], '%Y-%m-%d %H:%M:%S')
date = datetime.date.strftime(d, "%d/%m/%y %H:%M")
print (date)

line.update({'date':date})

df = pd.read_csv('apple_1_test.csv')
df = df.append(line, ignore_index=True)
df.to_csv('apple_1_test.csv')


'''from csv import DictWriter
 
def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)'''
        
        

# This bit of code will write the result of the query to output.csv

with open('output.csv', 'a') as f:
    f.write(r.text)