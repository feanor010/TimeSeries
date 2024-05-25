import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("1_model.csv")

date_columns = [col for col in df.columns if 'date_tag' in col]
value_columns = [col for col in df.columns if 'value_tag' in col]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])


df.dropna(subset=date_columns + value_columns, inplace=True)

data = pd.DataFrame()

for i, col in enumerate(date_columns):
    data[col] = df[col]
    data[f'Value_{i}'] = df[value_columns[i]]

data.set_index(date_columns[0], inplace=True)

data.interpolate(method='ffill', inplace=True)

num_plots = len(date_columns)

num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7*num_rows))

for i in range(num_plots):
    y = data[f'Value_{i}']

    train = y[:'2022-03-15']
    test = y['2022-03-15':]

    model = SARIMAX(train, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()

    forecast_steps = len(test)
    y_pred = model_fit.forecast(steps=forecast_steps)

    row = i // num_cols
    col = i % num_cols

    axs[row, col].plot(train.index, train, label='Train')
    axs[row, col].plot(test.index, test, label='Test')
    axs[row, col].plot(test.index, y_pred, color='darkgreen', label='Predictions')
    axs[row, col].set_title(f'Value_{i}')
    axs[row, col].legend()

for i in range(num_plots, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axs[row, col].remove()

plt.tight_layout()
plt.show()
