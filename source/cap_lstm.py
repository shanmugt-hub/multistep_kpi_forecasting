# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:41:39 2026

@author: Shanmuganathan T

Description:
------------
This script implements:
1. Single-step LSTM forecasting for KPI time-series
2. Residual diagnostics (distribution + ACF)
3. Recursive multi-step forecasting (12-hour horizon)
4. Multi-step evaluation metrics

The model captures short-term temporal dependencies using a sliding
window approach (window = 24, representing daily seasonality).
"""
# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# 2. Load Dataset
# ===============================
# C:Personal\\Walsh\\DA_Capstone\\dataset\\kpi_metric_1001_v1.csv
dir_path = "C:\\Personal\\Walsh\\DA_Capstone\\dataset\\" 
filename = "kpi_output.csv"
file_path = os.path.join(dir_path, filename)

df = pd.read_csv(file_path, header=0)
# Rename columns for consistency
df.columns = ["timestamp", "value"]
# Convert timestamp column to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])
# Set timestamp as index for time-series analysis
df.set_index("timestamp",inplace=True)
# Basic EDA checks
df.head()
df.info()
df.describe()
df.isnull().sum()
df.shape
df.columns
df.dtypes

# ===============================
# 3. Normalisation (Scaling)
# ===============================
# LSTM performs better when data is scaled.
# MinMaxScaler scales values between 0 and 1.
## Normalise the timeseries values
series = df["value"].values.reshape(-1,1)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# ===============================
# 4. Create Sliding Window Sequences
# ===============================
# This converts the time-series into supervised learning format.
# Each input contains 'window' past observations,
# and the target is the next time step.

## Single-step forecast capturing daily pattern of window 24
def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])   # past 24 observations
        y.append(data[i+window])     # next observation
    return np.array(X), np.array(y)

window = 24 # 24-hour daily seasonality
X, y = create_sequences(series_scaled, window)

X.shape
y.shape


# ===============================
# 5. Train-Test Split (80-20)
# ===============================

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================
# 6. Build LSTM Model
# ===============================
# Single LSTM layer with 64 units.
# Dense(1) outputs single-step forecast.

model = Sequential()
model.add(LSTM(64, input_shape=(window,1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.summary()
#del model
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)


# ===============================
# 7. Single-Step Prediction
# ===============================
# Predict on test data
pred_scaled = model.predict(X_test)
# Convert predictions back to original scale
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test)


# ===============================
# 8. Evaluation Metrics
# ===============================
mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100

print("LSTM MAE:", mae)
print("LSTM RMSE:", rmse)
print("LSTM MAPE:", mape)

# ===============================
# 9. Plot Actual vs Prediction
# ===============================
#Plot the actuals and predictions
plt.figure(figsize=(12,6))
plt.plot(df.index[-len(actual):],actual, label="Actual", marker='o')
plt.plot(df.index[-len(pred):],pred, label="Prediction", marker='x', linestyle="--")
plt.title("LSTM forecast: Actual vs Prediction")
plt.xlabel("Time Steps")
plt.ylabel("KPI Value")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 10. Residual Analysis
# ===============================

residuals = actual.flatten() - pred.flatten()
plt.figure(figsize=(12,6))
plt.plot(df.index[-len(actual):],residuals)
plt.title("LSTM Residuals")
plt.show()

plt.figure()
plt.hist(residuals, bins=30)
plt.title("LSTM Residual Distribution")
plt.show()

# ===============================
# 11. Autocorrelation of Residuals
# ===============================
# If residuals still show autocorrelation,
# model has not captured all temporal structure.
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(residuals, lags=48)
plt.title("LSTM Residual ACF")
plt.show()


# ===============================
# 12. Recursive Multi-Step Forecasting
# ===============================
# This predicts multiple future time steps
# by feeding previous predictions back into the model.

# Recursive multi-step LSTM forecasting , 12 Hrs in future
# Rolling walk forecast

def multi_step_forecast(model, last_window, horizon):
    forecast = []
    current_window = last_window.copy()

    for _ in range(horizon):
        pred = model.predict(current_window.reshape(1, window, 1), verbose=0)
        forecast.append(pred[0,0])

        current_window = np.append(current_window[1:], pred)

    return np.array(forecast)

horizon = 12 # Forecast 12 hours ahead

last_window = series_scaled[-window:]
forecast_scaled = multi_step_forecast(model, last_window, horizon)
forecast = scaler.inverse_transform(forecast_scaled.reshape(-1,1))

plt.figure(figsize=(12,6))
plt.plot(df.index[-100:], df["value"].values[-100:], label="Recent KPI")
plt.plot(
    pd.date_range(df.index[-1], periods=horizon+1, freq="H")[1:],
    forecast,
    label="LSTM Multi-step Forecast"
)
plt.legend()
plt.title("Multi-step LSTM Forecast")
plt.show()

# ===============================
# 13. Multi-Step Evaluation (OPTIONAL)
# ===============================

steps = 24

last_window = series_scaled[-(steps + window):-steps]
forecast_scaled = multi_step_forecast(model, last_window, steps)
forecast = scaler.inverse_transform(forecast_scaled.reshape(-1,1))
actual = series[-steps:]

mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
mape = np.mean(np.abs((actual - forecast) / actual)) * 100

print("Multi-step MAE:", mae)
print("Multi-step RMSE:", rmse)
print("Multi-step MAPE:", mape)

plt.figure(figsize=(12,6))
plt.plot(df.index[-len(actual):],actual, label="Actual")
plt.plot(df.index[-len(actual):],forecast, label="Forecast")
plt.legend()
plt.title("Multi-step LSTM Forecast vs Actual")
plt.show()


####### ========================================== #######

