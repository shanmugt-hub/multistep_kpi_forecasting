# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:21:43 2026

@author: Shanmuganathan T
"""

#KPI Multistep Forecasting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns

warnings.filterwarnings("ignore")

#Time series KPI Multistep Forecasting

# ------------------------------
# Load Dataset
# ------------------------------

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
# Data Preprocessing
# ------------------------------

df = df_raw.sort_values(by='timestamp')
# Drop unnecessary columns
df = df.drop(columns=["kpi_id", "entity_id"])

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
# Identify duplicate timestamps
# ------------------------------
duplicate_rows = df[df.index.duplicated(keep=False)]

# Count duplicates
num_duplicates = duplicate_rows.shape[0]
num_unique_duplicate_timestamps = duplicate_rows.index.nunique()

print(f"Number of rows with duplicate timestamps: {num_duplicates}")
print(f"Number of duplicated timestamps: {num_unique_duplicate_timestamps}")

# Display duplicate timestamps and values
duplicate_rows.sort_values("timestamp").head()

df = df.groupby(df.index).mean()

df.isnull().sum()
df.shape

# ------------------------------
# Handle Missing Timestamps
# ------------------------------

# Create continuous hourly index
last_timestamp = df.index.max()
end_time = last_timestamp.normalize() + pd.Timedelta(hours=23)

full_index = pd.date_range(start=df.index.min(), end=end_time, freq="h")

# Reindex to full hourly timeline
df = df.reindex(full_index)
# Count of NAN KPI values before imputation for the missing timestamps
df.isnull().sum()
df.shape # WE have 720 data points now

df_clean = df.copy(deep=True)
df_clean.isnull().sum()
# Impute missing KPI values
df_clean["value"] = df_clean["value"].ffill().bfill()

# Count of NAN KPI values after imputation
df_clean.isnull().sum()

filename = "kpi_output.csv"
output_path = os.path.join(dir_path, filename)
df_clean.to_csv(output_path, index=True)

# ------------------------------
# EDA
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

# Hourly average data
df_hourly = df_clean['value'].resample('H').mean()

# Daily average
df_daily = df_clean['value'].resample('D').mean()

# Weekly average
df_weekly = df_clean['value'].resample('W').mean()

df_daily.plot(title="Daily Average", figsize=(15,4))
plt.grid(True)
plt.show()

df_weekly.plot(title="Weekly Average", figsize=(15,4), color='black')
plt.grid(True)
plt.show()

# ------------------------------
# Plot Hourly daily Patterns
# ------------------------------

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
# Seasonal Decomposition
# ------------------------------

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df_clean['value'], model='additive', period=24)
decomp.plot()
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


# ---------------------------------
### ADFuller Test for Stationarity
# ---------------------------------

#Stationarity Check (ADF Test)

from statsmodels.tsa.stattools import adfuller

result = adfuller(df_clean['value'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] < 0.05:
    print("Time series is stationary.")
else:
    print("Time series is non-stationary.")

###
#p-value < 0.5, Reject Null Hypothesis, Accept Alternative Hypothesis
###


# ------------------------------
# ARIMA Model Training
# ------------------------------

# Since the series is stationary, d = 0

from statsmodels.tsa.arima.model import ARIMA

#model_ar = ARIMA(df_clean["value"], order=(2, 0, 2))
#model_ar = ARIMA(df_clean["value"], order=(3, 1, 3))
model_ar = ARIMA(df_clean["value"], order=(3, 0, 5))
model_ar_fit = model_ar.fit()

print(model_ar_fit.summary())

# --------------------------------
# Forecasting with order=(3, 0, 5)
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

plt.figure()
plt.plot(df_clean.index, df_clean["value"], label="Observed")
plt.plot(forecast_index, forecast, label="Forecast", linestyle="--")
plt.legend()
plt.title(f"ARIMA Forecast (Next {forecast_horizon} Hours)")
plt.xlabel("Time")
plt.ylabel("KPI Value")
plt.show()

# ------------------------------
# Output Forecast Values
# ------------------------------
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
plt.plot(forecast_df_h.index, forecast_df_h['forecast'], label='f"{forecast_horizon}-Hour Forecast"', color='red')
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



# ----------------------------------------------
# Rolling Forecast with ARIMA (Cross-Validation)
# -----------------------------------------------

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

train_size = int(len(df_hourly) * 0.8)
train, test = df_hourly[:train_size], df_hourly[train_size:]

history = train.tolist()
predictions = []


# p=3, d=0, q=5 from ACF/PACF

for t in range(len(test)):
    model = ARIMA(history, order=(3,0,5))  
    model_fit = model.fit()
    output = model_fit.forecast()
    predictions.append(output[0])
    history.append(test[t])


mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))
mape = mean_absolute_percentage_error(test, predictions) * 100

print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAPE = {mape:.2f}%")


plt.figure(figsize=(12,6))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Hour')
plt.ylabel('KPI Value')
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------
# Manual Auto ARIMA (Grid Search with statsmodels)
# ------------------------------------------------

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

