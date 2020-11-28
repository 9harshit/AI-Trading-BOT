import random
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tqdm import tqdm


class AI_Trader():
    def __init__(self, state_size, action_space = 3): #HOLD, BUY, SELL

        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen = 2000) #EXPERIENCE REPLACY
        self.inventory = [] #STOCKS WE OWN

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
        regressor.add(LSTM(units = 75, return_sequences = True, input_shape = self.state_size))
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





episodes = 10
batch_size = 32
timestep = 12
dataset_train = pd.read_csv('apple_5min.csv')
dataset_train = dataset_train.dropna()
training_set = dataset_train.iloc[:,1:].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 DAYsteps and 1 output
X_train = []
y_train = []
for i in range(timestep, (training_set_scaled.shape[0])):
    X_train.append(training_set_scaled[i-timestep:i])
X_train = np.array(X_train)

prev_profit = 0

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

data_samples = len(X_train) - 1

trader = AI_Trader((X_train.shape[1],5))

for episode in range(1, episodes+1):
    print("\nEpisodes :{}/{}".format(episode,episodes))
    total_porfit = 0
    trader.inventory = []
    trader.memory.clear()
    tran = 0

    state = X_train[0:1,:]



    for t in tqdm(range(data_samples)):
        action = trader.trade(state)
        next_state = X_train[t+1:t+2,:]

        actual_state_value = sc.inverse_transform(state[0:1,-1,:])

        reward = 0

        if action == 1 :
            trader.inventory.append(actual_state_value[-1,3])
            tran += 1
            print("\nApple Stock Bought at " + (str(actual_state_value[-1,3])))

        elif action == 2 and len(trader.inventory) > 0:
            tran += 1

            #Selling one stock at a time

            #buy_price = trader.inventory.pop(0)
            #reward = actual_state_value[-1,3] - buy_price
            #total_porfit += actual_state_value[-1,3] - buy_price
            #print("\nApple Stock Bought at " + str(buy_price)+ " Sold at " + str(actual_state_value[-1,3]),"Profit " + str(actual_state_value[-1,3] - buy_price))

            #Selling all stocks present in the inventory

            value = [x - actual_state_value[-1,2] for x in trader.inventory]
            reward += sum(value)
            total_porfit += sum(value)
            trader.inventory.clear()
            print("\nProfit on the trade", sum(value))


        else :
            print("\nHolding Apple Stock")
        if t == data_samples - 1:
            done = True
        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))
        state = next_state

        print("Trans :", tran)
        print("\nTOTAL PROFIT: {}\n".format((total_porfit)))



        if done:
            if len(trader.inventory)>0:
                #Selling remaining of stock left after completion of the episode
                value = [x - actual_state_value[-1,2] for x in trader.inventory]
                total_porfit += sum(value)
            trader.model.save("trader_rnn_5min.h5")
            print("***Episode done***")

        if len(trader.memory) > batch_size:
            print("batch trade")
            trader.batch_trade(batch_size)

    print("***Training done***")
