import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf 

from tqdm import tqdm
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

        #MODEL WILL RETURN WILL ACTION TO PERFROM AND INPUT IS ---

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units = 32, activation = 'relu', input_dim = self.state_size))
        model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(units = self.action_space, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 0.001)) #OUTPUT AS SELL, BUY, HOLD

        return model

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
            self.model.fit(state, target, epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

#TO GET REAL JUMP IN PRICES EVEN THOUGH PRICES ARE DIFFERNT
def sigmoid(x):
    return 1/ (1 + math.exp(-x))

def stock_price_format(n):
    if n < 0:
        return "- ${:.2f}".format(abs(n))

    else:
        return "${:.2f}".format(abs(n))

def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size +1

    if starting_id >= 0:
        windowed_data = data[starting_id:timestep+1]
    else:
        windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])

    state = []

    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))

    return np.array([state])

dataset = pd.read_csv("apple_1min.csv")
data = np.array(dataset["Close"])
episodes = 100
batch_size = 32
data_samples  = len(data) - 1 
window_size = 6

trader = AI_Trader(window_size)

for episode in range(1, episodes+1):
    print("\nEpisodes :{}/{}".format(episode,episodes))
    total_porfit = 0 
    trader.inventory = []

    state = state_creator(data, 0, window_size+1)

    for t in tqdm(range(data_samples)):
        
        action = trader.trade(state)
        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0

        if action == 1 : 
            trader.inventory.append(data[t])

        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)

            reward = max(data[t] - buy_price, 0)
            total_porfit += data[t] - buy_price
            print("Sold at ",(stock_price_format(data[t])), "Profit",stock_price_format(data[t] - buy_price))

        else:
            print("Holding stock")
        if t == data_samples - 1 :
            done = True
        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("TOTAL PROFIT: {}".format(stock_price_format(total_porfit)))
            
        if len(trader.memory) > batch_size:
           trader.batch_trade(batch_size)
    if episode % 10 == 0:
        trader.model.save("traderann.h5")