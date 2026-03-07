# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:19:08 2026

@author: Shanmuganathan T

Description:
------------
This script implements a SARIMA + LSTM hybrid forecasting model
for KPI time-series prediction.

Methodology:
1. SARIMA models linear and seasonal components.
2. Residuals from SARIMA are extracted.
3. LSTM is trained on SARIMA residuals to capture nonlinear patterns.
4. Final Hybrid Forecast = SARIMA Forecast + LSTM Residual Forecast.

"""
# ===============================
# 1. Import Required Libraries
# ===============================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. Load and Prepare Dataset
# ===============================
# Load the dataset from local directory
# C:Personal\\Walsh\\DA_Capstone\\dataset\\kpi_metric_1001_v1.csv
dir_path = "C:\\Personal\\Walsh\\DA_Capstone\\dataset\\" 
filename = "kpi_output.csv"
file_path = os.path.join(dir_path, filename)

df = pd.read_csv(file_path, header=0)
df.columns = ["timestamp", "value"]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp",inplace=True) # Set timestamp as index for time-series modeling
series = df["value"]  # Extract univariate time-series
# Basic exploratory checks
df.head()
df.info()
df.describe()
df.isnull().sum()
df.shape
df.columns
df.dtypes

# ===============================
# 3. Train-Test Split (80-20)
# ===============================
# First 80% used for training
# Remaining 20% used for evaluation

train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]


# ===============================
# 4. SARIMA Model (Linear + Seasonal)
# ===============================
# order = (p,d,q)
# seasonal_order = (P,D,Q,s)
# s=24 captures daily seasonality for hourly KPI data

sarima_model = SARIMAX(
    train,
    order=(1,0,1),
    seasonal_order=(1,1,1,24),  # change seasonality if daily/weekly
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_result = sarima_model.fit()

# SARIMA Forecast
sarima_forecast = sarima_result.forecast(steps=len(test))

# ===============================
# 5. Compute SARIMA Residuals
# ===============================
# Residual = Actual - Fitted
# These contain nonlinear components not captured by SARIMA

train_fitted = sarima_result.fittedvalues
residuals = train - train_fitted
# Drop NaNs introduced due to differencing
residuals = residuals.dropna()


# ===============================
# 6. Prepare Residuals for LSTM
# ===============================
# Scale residuals for neural network training

scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1,1))

# Convert residual time-series into supervised learning format
# Each input contains previous 24 residuals
# Target is next residual

def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window = 24
X_res, y_res = create_sequences(residuals_scaled, window)
X_res.shape
y_res.shape

# Reshape to 3D for LSTM: (samples, timesteps, features)
X_res = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))


# ===============================
# 7. Build LSTM Model (Nonlinear Component)
# ===============================
# LSTM learns nonlinear patterns in residual series

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(window,1)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

lstm_model.fit(
    X_res, y_res,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("Train length:", len(train))
print("Test length:", len(test))
print("SARIMA forecast length:", len(sarima_forecast))
print("Residual length:", len(residuals))
print("Window size:", window)

# ===============================
# 8. Forecast Residuals using LSTM
# ===============================
# Recursive forecasting:
# Each predicted residual is fed back into the model
# to predict the next residual

last_window = residuals_scaled[-window:]
current_input = last_window.reshape((1, window, 1))

lstm_residual_forecast = []

for _ in range(len(test)):
    # Predict next residual
    pred = lstm_model.predict(current_input, verbose=0)
    lstm_residual_forecast.append(pred[0,0])
    
    # Reshape to maintain 3D structure
    pred_reshaped = pred.reshape((1,1,1))
    
    # Slide window forward
    current_input = np.concatenate(
        (current_input[:,1:,:], pred_reshaped),
        axis=1
    )

# Convert residual forecasts back to original scale
lstm_residual_forecast = scaler.inverse_transform(
    np.array(lstm_residual_forecast).reshape(-1,1)
).flatten()

print("lstm_residual_forecast size:", len(lstm_residual_forecast))

# ===============================
# 9. Hybrid Forecast
# ===============================
# Hybrid = SARIMA (linear) + LSTM (nonlinear residual)

hybrid_forecast = sarima_forecast.values + lstm_residual_forecast

# ===============================
# 10. Evaluation Metrics
# ===============================
# MAE  = Mean Absolute Error
# RMSE = Root Mean Squared Error
# MAPE = Mean Absolute Percentage Error

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# Evaluate SARIMA baseline
sarima_mae, sarima_rmse, sarima_mape = evaluate(test, sarima_forecast)
# Evaluate Hybrid model
hybrid_mae, hybrid_rmse, hybrid_mape = evaluate(test, hybrid_forecast)

print("===== SARIMA Performance =====")
print(f"MAE  : {sarima_mae:.4f}")
print(f"RMSE : {sarima_rmse:.4f}")
print(f"MAPE : {sarima_mape:.2f}%")

print("\n===== SARIMA + LSTM Hybrid Performance =====")
print(f"MAE  : {hybrid_mae:.4f}")
print(f"RMSE : {hybrid_rmse:.4f}")
print(f"MAPE : {hybrid_mape:.2f}%")

# ===============================
# 11. Visualization of Results
# ===============================
# Compare actual vs SARIMA vs Hybrid

plt.figure(figsize=(12,6))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, sarima_forecast, label="SARIMA")
plt.plot(test.index, hybrid_forecast, label="Hybrid")
plt.legend()
plt.title("SARIMA vs Hybrid Forecast")
plt.show()

####### ========================================== #######
