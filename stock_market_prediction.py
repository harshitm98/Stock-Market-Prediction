# -*- coding: utf-8 -*-
"""
Created on Wed May 30 23:39:38 2018

@author: Harshit Maheshwari
"""

# Stock market prediction


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Nifty_minute_data/banknifty.csv')
X = dataset.iloc[:,3:6].fillna(method = 'pad').values
y = dataset.iloc[:, 6].fillna(method = 'pad').values
X = X[:-1, :]
y = y[1:]
length_train = int(len(X)*0.8)

# Splitting the data
X_train = X[:length_train, :]
y_train = y[:length_train]
X_test = X[length_train:, :]
y_test = y[length_train:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 30)
regressor.fit(X_train , y_train)

# Predicting the values
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
#y_test = sc_y.transform(y_test)

# Visualizing the results
plt.plot(y_test, color = 'blue')
plt.plot(y_pred, color = 'red')
plt.xlabel('Time stamp')
plt.ylabel('Value of the stock')
plt.title('Stock Prediction for Bank Nifty')
plt.show()