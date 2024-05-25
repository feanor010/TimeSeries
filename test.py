import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("1_model.csv")

date_columns = [col for col in df.columns if 'date_tag' in col]
value_columns = [col for col in df.columns if 'value_tag' in col]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

df.dropna(subset=date_columns + value_columns, inplace=True)

df.interpolate(method='ffill', inplace=True)

df[value_columns] = df[value_columns].apply(pd.to_numeric, errors='coerce')

df.dropna(inplace=True)

seq_length = 10

X = []
y = []

for i in range(len(df) - seq_length):
    X.append(df[value_columns].iloc[i:i+seq_length].values)
    y.append(df[value_columns].iloc[i+seq_length].values)

X = np.array(X)
y = np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    SimpleRNN(64),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)

num_plots = len(value_columns)

num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7*num_rows))

for i in range(num_plots):
    row = i // num_cols
    col = i % num_cols

    axs[row, col].plot(df[date_columns[-len(X_test):]].iloc[:, 0][:len(y_test)], y_test[:, i], label='Actual')
    axs[row, col].plot(df[date_columns[-len(X_test):]].iloc[:, 0][:len(y_test)], y_pred[:, i], label='Predicted', color='darkgreen')
    axs[row, col].set_xlabel('Time')
    axs[row, col].set_ylabel('Value')
    axs[row, col].set_title(f'Time Series {i+1}')
    axs[row, col].legend()

plt.tight_layout()
plt.show()
