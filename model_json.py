#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:38:28 2020

@author: harshit
"""



from keras.models import model_from_json

model_json = regressor1.to_json()
with open("regressor1.json", "w") as json_file:
    json_file.write(model_json)
 
 
# load json and create model
json_file = open('regressor1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor1 = model_from_json(loaded_model_json)
# load weights into new model
regressor1.load_weights("biapple1.h5")
 

# Compiling the RNN
regressor1.compile(optimizer = 'adam', loss = 'mean_squared_error')




