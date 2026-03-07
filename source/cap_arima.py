# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:21:43 2026

@author: Shanmuganathan T

Title:
------
KPI Multi-step Time Series Forecasting

Description:
------------
This script performs:
1. Data preprocessing and cleaning
2. Exploratory Data Analysis (EDA)
3. Stationarity testing (ADF, KPSS)
4. ARIMA modeling
5. Rolling walk-forward validation
6. SARIMA modeling
7. Manual Auto-ARIMA (Grid Search)
8. Multi-step forecasting with confidence intervals

The goal is to forecast hourly KPI values using classical
time-series techniques.
"""
# ===============================
# 1. Import Required Libraries
# ===============================

#KPI Multistep Forecasting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns

warnings.filterwarnings("ignore")

#Time series KPI Multistep Forecasting

# ===============================
# 2. Load Dataset
# ===============================

# Load the dataset from local directory
# C:Personal\\Walsh\\DA_Capstone\\dataset\\kpi_metric_1001_v1.csv
file_path = "C:\\Personal\\Walsh\\DA_Capstone\\dataset\\kpi_metric_1001_v1.csv" 
dir_path = "C:\\Personal\\Walsh\\DA_Capstone\\dataset\\" 

df_raw = pd.read_csv(file_path, header=0)


df_raw.head()
df_raw.info()
df_raw.describe()
df_raw.isnull().sum()
df_raw.shape
df_raw.columns
df_raw.dtypes

# ------------------------------
# 3. Data Preprocessing
# ------------------------------

df = df_raw.sort_values(by='timestamp')
# Drop unnecessary columns
df = df.drop(columns=["kpi_id", "entity_id"])
# Convert timestamp and KPI values to correct types
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d-%m-%Y %H:%M", dayfirst=True, errors="coerce"
)
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Drop invalid timestamps
df = df.dropna(subset=["timestamp"])

# Set timestamp as index
df = df.set_index("timestamp").sort_index()

# Handle duplicate timestamps by averaging
# ------------------------------
# 4. Handle duplicate timestamps
# ------------------------------

# Identify duplicate timestamps
duplicate_rows = df[df.index.duplicated(keep=False)]

# Count duplicates
num_duplicates = duplicate_rows.shape[0]
num_unique_duplicate_timestamps = duplicate_rows.index.nunique()

print(f"Number of rows with duplicate timestamps: {num_duplicates}")
print(f"Number of duplicated timestamps: {num_unique_duplicate_timestamps}")

# Display duplicate timestamps and values
duplicate_rows.sort_values("timestamp").head()

# Aggregate duplicates by averaging values
df = df.groupby(df.index).mean()

df.isnull().sum()
df.shape

# ------------------------------
# 5. Handle Missing Timestamps
# ------------------------------

# Create continuous hourly index
last_timestamp = df.index.max()
end_time = last_timestamp.normalize() + pd.Timedelta(hours=23)

full_index = pd.date_range(start=df.index.min(), end=end_time, freq="h")

# Reindex to continuous hourly timeline
df = df.reindex(full_index)
# Count of NAN KPI values before imputation for the missing timestamps
df.isnull().sum()
df.shape # WE have 720 data points now

df_clean = df.copy(deep=True)
df_clean.isnull().sum()
# Impute missing values using forward-fill + backward-fill
df_clean["value"] = df_clean["value"].ffill().bfill()

# Count of NAN KPI values after imputation
df_clean.isnull().sum()
# Export cleaned dataset for future use
filename = "kpi_output_1.csv"
output_path = os.path.join(dir_path, filename)
df_clean.to_csv(output_path, index=True)

# ------------------------------
# 6. EDA
# ------------------------------

#Box plot
plt.figure(figsize=(10, 4))
sns.boxplot(data=df_clean, x='value', color='skyblue')
plt.title("Box Plot of KPI value")
#plt.xlabel("KPI Value distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['value'], bins=50, kde=True, color='mediumseagreen')
plt.title("Distribution of KPI Values")
#plt.xlabel("DL Throughput (Mbps)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# PLOT BEFORE vs AFTER IMPUTATION
# -------------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df["value"],
    label="Before Imputation",
    linewidth=1.5
)

plt.plot(
    df_clean.index,
    df_clean["value"],
    label="After Imputation",
    linestyle="--",
    linewidth=1.5
)

plt.title("KPI Values Before and After Imputation")
plt.xlabel("Time")
plt.ylabel("KPI Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# -------------------------------
# Aggregated temporal patterns
# -------------------------------
# Hourly average data
df_hourly = df_clean['value'].resample('H').mean()

# Daily average
df_daily = df_clean['value'].resample('D').mean()

# Weekly average
df_weekly = df_clean['value'].resample('W').mean()

df_hourly.plot(title="Hourly Average", figsize=(15,4))
plt.grid(True)
plt.show()

df_daily.plot(title="Daily Average", figsize=(15,4))
plt.grid(True)
plt.show()

df_weekly.plot(title="Weekly Average", figsize=(15,4), color='black')
plt.grid(True)
plt.show()

# ------------------------------------------
# Plot hour-of-day day-of-the-week Patterns
# ------------------------------------------

df_clean['hour'] = df_clean.index.hour
df_clean['dayofweek'] = df_clean.index.dayofweek

plt.figure(figsize=(12,6))
sns.boxplot(data=df_clean, x='hour', y='value')
plt.title("Distribution by Hour of Day")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df_clean, x='dayofweek', y='value')
plt.title("Distribution by Day of Week")
plt.xticks(ticks=range(7), labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.show()

df_clean['rolling_mean_24h'] = df_clean['value'].rolling(window=24).mean()  
df_clean[['value', 'rolling_mean_24h']].plot(figsize=(15,5), title="Original vs 24h Rolling Mean")
plt.grid(True)
plt.show()

# ------------------------------
# 7. Seasonal Decomposition
# ------------------------------

from statsmodels.tsa.seasonal import seasonal_decompose
# Decompose into trend, seasonal, residual components
decomp = seasonal_decompose(df_clean['value'], model='additive', period=24)
decomp.plot()

full_residues = decomp.resid
df_clean['residues'] = full_residues
df_clean['residues'] = df_clean['residues'].dropna()

plt.show()

# ------------------------------
# ACF and PACF Plot
# ------------------------------


from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

autocorrelation_plot(df_clean['value'])
plt.title("Autocorrelation - ACF")
plt.show()

plot_acf(df_clean['value'], lags=100)
plt.show()

plot_pacf(df_clean['value'], lags=50)
plt.show()

# ------------------------------
# 8. Stationarity Tests
# ------------------------------

# Stationarity Check (ADF Test)
###
# p-value < 0.5, Reject Null Hypothesis, Accept Alternative Hypothesis
# H0 - non-stationary, H1 - Stationary
###

from statsmodels.tsa.stattools import adfuller

result = adfuller(df_clean['value'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] < 0.05:
    print("Time series is stationary.")
else:
    print("Time series is non-stationary.")


# Stationarity Check (KPSS)
###
# p-value < 0.5, Reject Null Hypothesis, Accept Alternative Hypothesis
# H0 - Stationary, H1 - Non-stationary
###

from statsmodels.tsa.stattools import kpss
kpss_result = kpss(df_clean['value'].dropna(), regression="ct", nlags="auto")

print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])
print("Lags used:", kpss_result[2])
print("Critical Values:")
for key, val in kpss_result[3].items():
    print(f"   {key}: {val}")
    
if kpss_result[1] > 0.05:
    print("Time series is stationary.")
else:
    print("Time series is non-stationary.")


# ------------------------------
# 9. ARIMA Model Training
# ------------------------------

# Since the series is stationary, d = 0

from statsmodels.tsa.arima.model import ARIMA

model_ar = ARIMA(df_clean["value"], order=(1, 0, 1))
#model_ar = ARIMA(df_clean["value"], order=(3, 0, 5))
model_ar_fit = model_ar.fit()

print(model_ar_fit.summary())

# --------------------------------
# Forecasting with order=(1, 0, 1)
# --------------------------------

forecast_horizon = 12
forecast = model_ar_fit.forecast(steps=forecast_horizon)

forecast_index = pd.date_range(
    start=df_clean.index[-1],
    periods=forecast_horizon + 1,
    freq="H"
)[1:]

# ------------------------------
# Plot Forecast
# ------------------------------

plt.figure(figsize=(12,6))
#plt.plot(df_clean.index, df_clean["value"], label="Observed")
plt.plot(df_clean.index[-96:], df_clean["value"].iloc[-96:], label="Observed")

plt.plot(forecast_index, forecast, label="Forecast", linestyle="--")
plt.legend()
plt.title(f"ARIMA Forecast (Next {forecast_horizon} Hours)")
plt.xlabel("Time")
plt.ylabel("KPI Value")
plt.grid(True,linestyle="--", alpha=0.7)
plt.gcf().autofmt_xdate()
plt.tight_layout()


plt.show()

# ---------------------------------------------
# Output Multi-step Forecast Values with CI
# ---------------------------------------------

forecast_df = pd.DataFrame({
    "timestamp": forecast_index,
    "forecast_value": forecast.values
})

print(forecast_df)

# plot with confidence interval
forecast_result_h = model_ar_fit.get_forecast(steps=forecast_horizon, alpha=0.05)

forecast_mean_h = forecast_result_h.predicted_mean
conf_int_h = forecast_result_h.conf_int()

# Create future timestamps
future_index_h = pd.date_range(start=df_hourly.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')

forecast_df_h = pd.DataFrame({
    'forecast': forecast_mean_h,
    'lower_ci': conf_int_h.iloc[:, 0],
    'upper_ci': conf_int_h.iloc[:, 1]
}, index=future_index_h)


plt.figure(figsize=(12, 5))
plt.plot(df_hourly[-96:], label='Historical', color='black')
plt.plot(forecast_df_h.index, forecast_df_h['forecast'], label=f"{forecast_horizon}-Hour Forecast", color='red')
plt.fill_between(forecast_df_h.index,
                 forecast_df_h['lower_ci'],
                 forecast_df_h['upper_ci'],
                 color='blue', alpha=0.1, label='95% Confidence Interval')

plt.title(f"{forecast_horizon}-Hour KPI Forecast")
plt.xlabel("Time")
plt.ylabel("KPI Value")
plt.legend()
plt.grid(True)
plt.show()
print(forecast_df_h)


# ------------------------------------------------------------
# Multi-step Rolling Walk Froward Forecast with ARIMA - Model 1
# ------------------------------------------------------------

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

train = []
test = []

train_size = int(len(df_hourly) * 0.8)
train, test = df_hourly[:train_size], df_hourly[train_size:]

history = train.tolist()
predictions = []
residuals = []


# p=3, d=0, q=5 from ACF/PACF

for t in range(len(test)):
    #model_1 = ARIMA(history, order=(3,0,5))  
    model_1 = ARIMA(history, order=(1,0,1))  
    model_fit_1 = model_1.fit()
    output = model_fit_1.forecast()
    predictions.append(output[0])
    history.append(test[t])
    error = test[t] - output
    residuals.append(error)


residuals = np.array(residuals)
mae_1 = mean_absolute_error(test, predictions)
rmse_1 = np.sqrt(mean_squared_error(test, predictions))
mape_1 = mean_absolute_percentage_error(test, predictions) * 100

print(f"MAE  = {mae_1:.2f}")
print(f"RMSE = {rmse_1:.2f}")
print(f"MAPE = {mape_1:.2f}%")


plt.figure(figsize=(12,6))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Hour')
plt.ylabel('KPI Value')
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------
# Residual time-series plot
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(test.index, residuals)
plt.title("Rolling ARIMA Forecast Residuals")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.show()

# ---------------------------
# Residual histogram
# ---------------------------
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

'''
model_1 = ARIMA(train, order=(1,0,1))  
model_fit_1 = model_1.fit()
ar_forecast = model_fit_1.forecast(steps=len(test))

ar_mae = mean_absolute_error(test, ar_forecast)
ar_rmse = np.sqrt(mean_squared_error(test, ar_forecast))
ar_mape = mean_absolute_percentage_error(test, ar_forecast) * 100

print("MAE_ARIMA:", ar_mae)
print("RMSE_ARIMA:", ar_rmse)
print("MAPE_ARIMA:", ar_mape)

plt.figure(figsize=(12,6))
#plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, ar_forecast, label="Predicted", color='red')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Hour')
plt.ylabel('KPI Value')
plt.legend()
plt.grid(True)
plt.show()
'''


# ---------------------------------------------------------------------------
# Multi-step Rolling Forecast / ARIMA with extra steps - Model 2 ( Optional )
# ---------------------------------------------------------------------------
from statsmodels.tsa.arima.model import ARIMA

train_size = int(len(df_hourly) * 0.8)
train, test = df_hourly[:train_size], df_hourly[train_size:]

history = train.tolist()
predictions = []

extra_steps = 12
total_steps = len(test) + extra_steps

# Walk-forward forecast including future steps
for t in range(total_steps):
    #model_2 = ARIMA(history, order=(3,0,5))
    model_2 = ARIMA(history, order=(1,0,1))  
    model_fit_2 = model_2.fit()
    output = model_fit_2.forecast()
    predictions.append(output[0])
    
    if t < len(test):
        # For test steps, append the actual observed value
        history.append(test[t])
    else:
        # For future steps, append the forecast itself (since no actual data exists)
        history.append(output[0])

# Build extended index (test + 12 future hours)
future_index = pd.date_range(
    start=test.index[-1] + pd.Timedelta(hours=1),
    periods=extra_steps,
    freq="H"
)
extended_index = test.index.append(future_index)

mae = mean_absolute_error(test, predictions[:len(predictions)-extra_steps])
rmse = np.sqrt(mean_squared_error(test, predictions[:len(predictions)-extra_steps]))
mape = mean_absolute_percentage_error(test, predictions[:len(predictions)-extra_steps]) * 100

print(f"MAE_ARIMA  = {mae:.2f}")
print(f"RMSE_ARIMA = {rmse:.2f}")
print(f"MAPE_ARIMA = {mape:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(test.index, test.values, label="Observed", color="blue")
plt.plot(extended_index, predictions, label="Forecast (Test + 12h)", linestyle="--", color="orange")

plt.legend()
plt.title("ARIMA Walk-Forward Forecast Including 12 Future Hours")
plt.xlabel("Time")
plt.ylabel("KPI Value")
plt.grid(True, linestyle="--", alpha=0.7)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 11. SARIMA
# ------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX
train_size = int(len(df_hourly) * 0.8)
train, test = df_hourly[:train_size], df_hourly[train_size:]

sar_model = SARIMAX(
    train,
    order=(1,0,1),
    seasonal_order=(1,1,1,24),
    #enforce_stationarity=False,
    #enforce_invertibility=False
)

sar_results = sar_model.fit()

print(sar_results.summary())

sar_forecast = sar_results.forecast(steps=len(test))

sar_mae = mean_absolute_error(test, sar_forecast)
sar_rmse = np.sqrt(mean_squared_error(test, sar_forecast))
sar_mape = mean_absolute_percentage_error(test, sar_forecast) * 100

print("MAE_SARIMA:", sar_mae)
print("RMSE_SARIMA:", sar_rmse)
print("MAPE_SARIMA:", sar_mape)

plt.figure(figsize=(12,6))
#plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, sar_forecast, label="Predicted", color='red')
plt.title('SARIMA Forecast vs Actual')
plt.xlabel('Hour')
plt.ylabel('KPI Value')
plt.legend()
plt.grid(True)
plt.show()

# Extract residuals
residuals = sar_results.resid

plt.figure(figsize=(12,6))
plt.plot(residuals)
plt.title("SARIMA Residuals")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.show()



# ------------------------------------------------
# SARIMA - Rolling forecast
# ------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX

train = []
test = []

train_size = int(len(df_hourly) * 0.8)
train, test = df_hourly[:train_size], df_hourly[train_size:]

sar_history = train.tolist()
sar_predictions = []
sar_residuals = []

for t in range(len(test)):
    #model_1 = ARIMA(history, order=(3,0,5))  
    sar_model_1 = SARIMAX(sar_history, order=(1,0,1), seasonal_order=(1,1,1,24))  
    sar_model_fit_1 = sar_model_1.fit()
    sar_output = sar_model_fit_1.forecast()
    sar_predictions.append(sar_output[0])
    sar_history.append(test[t])
    sar_error = test[t] - sar_output
    sar_residuals.append(sar_error)


sar_residuals = np.array(sar_residuals)

sar_mae_1 = mean_absolute_error(test, sar_predictions)
sar_rmse_1 = np.sqrt(mean_squared_error(test, sar_predictions))
sar_mape_1 = mean_absolute_percentage_error(test, sar_predictions) * 100

print("MAE_SARIMA:", sar_mae_1)
print("RMSE_SARIMA:", sar_rmse_1)
print("MAPE_SARIMA:", sar_mape_1)

plt.figure(figsize=(12,6))
#plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, sar_predictions, label="Predicted", color='red')
plt.title('SARIMA Forecast vs Actual')
plt.xlabel('Hour')
plt.ylabel('KPI Value')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Residual time-series plot
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(test.index, sar_residuals)
plt.title("SARIMA Rolling Forecast Residuals")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.show()

# ---------------------------
# Residual histogram
# ---------------------------
plt.figure(figsize=(6,4))
plt.hist(sar_residuals, bins=30)
plt.title("SARIMA Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------------------------
# 12. Manual Auto ARIMA (Grid Search with statsmodels, OPTIONAL)
# ------------------------------------------------------------

def auto_arima_manual(ts, p_range=5, d_range=1, q_range=5):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(p_range + 1):
        for d in range(d_range + 1):
            for q in range(q_range + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, d, q)
                        best_model = model
                except(ValueError, np.linalg.LinAlgError):
                    continue

    print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
    return best_model, best_order, best_aic

best_model_h, best_order_h, best_aic_h = auto_arima_manual(df_hourly)
##best_model_d, best_order_d, best_aic_d = auto_arima_manual(df_daily)

# Forecast for 24 hours with auto-arima
forecast_result_h = best_model_h.get_forecast(steps=24, alpha=0.05)

forecast_mean_h = forecast_result_h.predicted_mean
conf_int_h = forecast_result_h.conf_int()

# Create future timestamps
future_index_h = pd.date_range(start=df_hourly.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

forecast_df_h = pd.DataFrame({
    'forecast': forecast_mean_h,
    'lower_ci': conf_int_h.iloc[:, 0],
    'upper_ci': conf_int_h.iloc[:, 1]
}, index=future_index_h)


plt.figure(figsize=(12, 5))
plt.plot(df_hourly[-24:], label='Historical', color='black')
plt.plot(forecast_df_h.index, forecast_df_h['forecast'], label='24-Hour Forecast', color='red')
plt.fill_between(forecast_df_h.index,
                 forecast_df_h['lower_ci'],
                 forecast_df_h['upper_ci'],
                 color='blue', alpha=0.1, label='95% Confidence Interval')

plt.title("24-Hour KPI Forecast")
plt.xlabel("Time")
plt.ylabel("KPI Value")
plt.legend()
plt.grid(True)
plt.show()
print(forecast_df_h)


####### ========================================== #######


