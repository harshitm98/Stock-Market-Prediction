# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 01:24:05 2018

@author: Harshit Maheshwari
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import time 

dataset = pd.read_csv('C:/Users/Ambika/Desktop/CS/Stock Market Prediction/Nifty_minute_data/banknifty.csv')
X = dataset.iloc[:,3:6].fillna(method = 'pad').values
y = dataset.iloc[:, 6].fillna(method = 'pad').values
X_1 = y[:-4][np.newaxis]
X_2 = y[1:-3][np.newaxis]
X_3 = y[2:-2][np.newaxis]
X = X[3:-1, :]
y = y[4:]
X  = np.append(X, X_1.T, axis = 1)
X = np.append(X, X_2.T, axis = 1)
X = np.append(X, X_3.T, axis = 1)
length_train = int(len(X)*0.8)

X_train = X[:length_train, :]
y_train = y[:length_train]
X_test = X[length_train:, :]
y_test = y[length_train:]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
# First hidden layer
model.add(LSTM(input_dim = 6,
              output_dim = 50,
              return_sequences = True))
model.add(Dropout(0.2))
#Second hidden layer
model.add(LSTM(100,
               return_sequences=False))
model.add(Dropout(0.5))
#Output layer
model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))
model.compile(loss= 'mse', optimizer = 'rmsprop')

model.fit(X_train,
         y_train,
         batch_size = 64,
         epochs = 30,
         validation_split = 0.05)

y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

plt.plot(y_test, color = 'blue', label = 'test')
plt.plot(y_pred, color = 'red', label = 'pred')
plt.xlabel('Time stamp')
plt.ylabel('Value of the stock')
plt.title('Stock Prediction for Bank Nifty with LSTM')
plt.legend()
plt.show()

plt.savefig('C:/Users/Ambika/Desktop/CS/Stock Market Prediction/stock_prediction_using_LSTM.png', dpi = 72)
