import math
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from tqdm import  tqdm
from collections import deque

class AI_Trader():
    def __init__(self, state_size, action_space = 3, model_name = "AITRADER"): #HOLD, BUY, SELL
       
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen = 2000) #EXPERIENCE REPLACY
        self.inventory = [] #STOCKS WE OWN
        self.model_name = model_name

        self.gamma = 0.90
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.990

        self.model = self.model_builder()

    def model_builder(self):

        #MODEL WILL RETURN WILL ACTION TO PERFROM AND INPUT IS State/prices
        
        # Initialising the RNN
        regressor = Sequential()
        
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 75, return_sequences = True, input_shape = (state_reshaped.shape[1], 5)))
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
        regressor.add(Dense(units = self.action_space, activation = "linear"))
        
        # Compiling the RNN
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')#OUTPUT AS SELL, BUY, HOLD

        return regressor

    #TRADE FUNCTION TAKES STATE AS AN INPUT AND OUTPUT IS ACTION TO PERFROM ACTION IN PARTICULAR STATE
    def trade(self, state):

        #ACTION SELECTION , SELECT ACTION FROM THE MODEL OR RANDOM ACTION

        if random.random() <= self.epsilon:
            return random.randrange(self.action_space) #returns random action 

        action = self.model.predict(state)
        #WILL RETURN ACTION WITH HIGHEST PROBABILITY
        return np.argmax(action[0]) #0 because of output shape

    #WILL TAKE A BATCH OF DATA AND WILL TRAIN MODEL ON THAT
    def batch_trade(self, batch_size):
        batch = []
        #SINCE TIMESERIES DATA TAKE ON LAST PART OF DATA
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward
            
            self.model.fit(state, target, epochs = 1, verbose = 0 )

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

#TO GET REAL JUMP IN PRICES EVEN THOUGH PRICES ARE DIFFERNT
def sigmoid(x):
    return 1/ (1 + math.exp(-x))

def stock_price_format(n):
    if n < 0:
        return "-${:.2f}".format(abs(n))

    else:
        return "${:.2f}".format(abs(n))

def state_creator(data, timestep, window_size):
    
    starting_id = timestep - window_size +1

    if starting_id >= 0:
        windowed_data = data[starting_id:timestep+1,:]
    else:
        windowed_data = - starting_id * [data[0,:]] + list(data[0:timestep+1,:])
        
    state = []
    
    for i in range(window_size - 1):
        state.append((windowed_data[i]))

    return np.array(state)

dataset_train = pd.read_csv('apple_5min.csv')
dataset_train = dataset_train.dropna()
training_set = dataset_train.iloc[:,1:].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


def reshaping_state(data, rnn_tmstp):
    state_reshaped = []
    
    for i in range(rnn_tmstp, (data.shape[0])):
        state_reshaped.append(data[i-rnn_tmstp:i])
        
    state_reshaped = np.array(state_reshaped)
    state_reshaped = np.reshape(state_reshaped, (state_reshaped.shape[0], state_reshaped.shape[1], 5))
    return state_reshaped

episodes = 10
batch_size = 32
data_samples  = len(training_set_scaled) - 1 
window_size = 65


state = state_creator(training_set_scaled, 0, window_size+1)

state_reshaped = reshaping_state(state, 60)
trader = AI_Trader(window_size)

for episode in range(1, episodes+1):
    print("\nEpisodes :{}/{}".format(episode,episodes))
    total_porfit = 0
    trader.inventory = []

    state = state_creator(training_set_scaled, 0, window_size+1)

    state_reshaped = reshaping_state(state, 60)
    
    for t in tqdm(range(data_samples)):
        
        action = trader.trade(state_reshaped)
        next_state = state_creator(training_set_scaled, t+1, window_size + 1)

        next_state_reshaped = reshaping_state(next_state, 60)

        reward = 0

        if action == 1 : 
            trader.inventory.append(training_set[t,3])
            print("\nApple Stock Bought at " + stock_price_format(training_set[t,3]))
                  
        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)

            reward = max(training_set[t,3] - buy_price, 0)
            total_porfit += training_set[t,3] - buy_price
            print("\nApple Stock Bought at " + stock_price_format(buy_price),"Sold at " + stock_price_format((training_set[t,3])),"Profit " + stock_price_format((training_set[t,3] - buy_price)))

        else :
            print("\nHolding Apple Stock")
        if t == data_samples - 1 :
            done = True
        else:
            done = False

        trader.memory.append((state_reshaped, action, reward, next_state_reshaped, done))
        state_reshaped = next_state_reshaped
        
        

        if done:
            print("\nTOTAL PROFIT: {}".format(stock_price_format(total_porfit)))
           # trader.model.save("trader1min.h5")
        if len(trader.memory) > batch_size:
            trader.batch_trade(batch_size)
        



