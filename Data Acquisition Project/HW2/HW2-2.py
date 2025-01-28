#Solution for question 2

import pandas as pd 
import numpy as np from statsmodels.tsa.arima.model 
import ARIMA from statsmodels.tsa.statespace.sarimax

data = pd.read_csv('data.csv', header=None)
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

arima_model = ARIMA(train_data, order=(1,1,1)).fit()
arima_preds = arima_model.forecast(steps=len(test_data))[0]
arima_mse = np.mean((test_data.values.flatten() - arima_preds)**2)
print("ARIMA MSE:", arima_mse)