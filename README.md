### PROJECT 4: TIME-SERIES-MODEL


**PRELIMINARIES** 

Time series analysis is a detailed way of analysing a sequence of data points collected over a period of time. To carry this analysis, analysts record data points at consistent intervals over a given period of time rather than just recording the data points at random or sporadically.
Time series data is pervasive in various fields like financial markets, economics, energy, healthcare, environmental sciences and many more.
To measure time series, it requires one to build a time series model that helps analyse and forecast the future. In this models, time is often the independent variable, and the goal is usually to make a prediction for the future. Understanding and effectively modeling time-dependent data is crucial for making informed decisions and predictions.
In this project, we will build a time series model Using the Craigslist Vehicles Dataset

The Data
We are using data from (https://www.kaggle.com/datasets/mbaabuharun/craigslist-vehicles)

```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

# uploading the dataset
tsm_data = pd.read_csv(r"C:\Users\SIMON.NGOTHO\Downloads\craigslist_vehicles.csv")

# Handling the missing values 
data['price'].fillna(data['price'].median(), inplace=True)
data['year'].fillna(data['year'].median(), inplace=True)
data['odometer'].fillna(data['odometer'].median(), inplace=True)
data['manufacturer'].fillna(data['manufacturer'].mode()[0], inplace=True)
data['model'].fillna(data['model'].mode()[0], inplace=True)
data['condition'].fillna(data['condition'].mode()[0], inplace=True)
data['cylinders'].fillna(data['cylinders'].mode()[0], inplace=True)
data['fuel'].fillna(data['fuel'].mode()[0], inplace=True)

tsm_data['posting_date'] = pd.to_datetime(tsm_data['posting_date'])

```

Converting to datetime to facilitate the analysis of temporal patterns

```python 
# Convert 'posting_date' to datetime 
tsm_data.set_index('posting_date', inplace=True)

```

Data visualization to help in analyzing complex data, identifying patterns, and extracting valuable insights. Simplifying complex information and presenting it visually enables the data users to make informed and effective decisions quickly and accurately.

```python 
# Data Exploration (Exploratory Data Analysis (EDA))
# visualization 

plt.figure(figsize=(14, 6))
plt.plot(tsm_data.index, tsm_data['price'], label='Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price - Time Series Plot')
plt.show()

```

Time Series Model is very instrumental as it can help organizations understand the underlying causes of trends or systemic patterns over time.

```python 
# Building a Time Series Model
price_series = tsm_data['price'].resample('M').mean()  # Resample to monthly data for smoothing
model = SimpleExpSmoothing(price_series).fit()
print(model.summary()) # check the model

# Building a Time Series Chart
plt.figure(figsize=(14, 6))
plt.plot(price_series.index, price_series, label='Price (Monthly Average)')
forecast = model.fittedvalues.append(model.forecast(steps=12))  # Forecast one year
forecast.plot(label='Forecast', color='yellow', edgecolor='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price - Time Series Forecast')
plt.legend()
plt.show()

```



