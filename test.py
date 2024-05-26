import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("1_model.csv")

date_columns = [col for col in df.columns if 'date_tag' in col]
value_columns = [col for col in df.columns if 'value_tag' in col]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

df.dropna(subset=date_columns + value_columns, inplace=True)
df.interpolate(method='ffill', inplace=True)
df[value_columns] = df[value_columns].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

scaler = MinMaxScaler()
df[value_columns] = scaler.fit_transform(df[value_columns])

seq_length = 50

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
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(y_train.shape[1])
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

X_pred = X_test[-seq_length:]
y_pred = model.predict(X_pred)

num_plots = len(value_columns)
num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7*num_rows))

for i in range(num_plots):
    row = i // num_cols
    col = i % num_cols

    real_indices = np.arange(len(y_test))
    pred_indices = np.arange(len(y_test) - seq_length, len(y_test))

    real_dates = df[date_columns[0]].iloc[split + real_indices].values
    pred_dates = df[date_columns[0]].iloc[split + pred_indices].values

    axs[row, col].plot(real_dates, y_test[:, i], label=f'Реальные {value_columns[i]}', color='blue', linestyle='-')
    axs[row, col].plot(pred_dates, y_pred[:, i], label=f'Предсказанные {value_columns[i]}', color='green', linestyle='--')
    axs[row, col].set_xlabel('Время')
    axs[row, col].set_ylabel('Значение')
    axs[row, col].set_title(f'Сравнение {value_columns[i]} за последние 10 дней')
    axs[row, col].legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss (обучение)')
plt.plot(history.history['val_loss'], label='Loss (валидация)')
plt.title('График обучения и валидации')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.show()
