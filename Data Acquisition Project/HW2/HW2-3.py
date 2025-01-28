#Solution for Question 3

import pandas as pd
import numpy as np from sklearn.metrics
import mean_squared_error from statsmodels.tsa.arima_model
import ARIMA from statsmodels.tsa.holtwinters

data = pd.read_csv('time_series.csv', header=None)
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

def arima(train, test, p, d, q):
    model = ARIMA(train, order=(p,d,q))
    model_fit = model.fit(disp=0)
    y_hat = test.copy()
    y_hat['ARIMA'] = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    mse = mean_squared_error(test, y_hat['ARIMA'])
    return y_hat, mse