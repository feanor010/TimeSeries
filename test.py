import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("test/3_model_test.csv")

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

seq_length = 5
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

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.05)

y_pred = model.predict(X_test)

y_train_denorm = scaler.inverse_transform(y_train)
y_test_denorm = scaler.inverse_transform(y_test)
y_pred_denorm = scaler.inverse_transform(y_pred)

metrics = {}
for i, col in enumerate(value_columns):
    mae = mean_absolute_error(y_test_denorm[:, i], y_pred_denorm[:, i])
    mse = mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i])
    metrics[col] = {'MAE': mae, 'MSE': mse}
    print(f'{col}: MAE = {mae:.4f}, MSE = {mse:.4f}')

num_plots = len(value_columns)
num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols
concat_array = np.vstack([y_train_denorm, y_test_denorm])

fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7*num_rows))

for i in range(num_plots):
    row = i // num_cols
    col = i % num_cols

    real_indices = np.arange(len(y_test) + len(y_train))
    pred_indices = np.arange(len(y_test) - len(y_pred), len(y_test))

    real_dates = df[date_columns[0]].iloc[real_indices].values
    pred_dates = df[date_columns[0]].iloc[split + pred_indices].values

    axs[row, col].plot(real_dates, concat_array[:, i], label=f'Реальные {value_columns[i]}', color='blue', linestyle='--')
    axs[row, col].plot(pred_dates, y_pred_denorm[:, i], label=f'Предсказанные {value_columns[i]}', color='red', linestyle='-')
    axs[row, col].set_xlabel('Время')
    axs[row, col].set_ylabel('Значение')
    axs[row, col].set_title(f'Сравнение {value_columns[i]} на всём тестовом периоде')
    axs[row, col].text(0.5, 0.95, f"MAE: {metrics[value_columns[i]]['MAE']:.4f}\nMSE: {metrics[value_columns[i]]['MSE']:.4f}",
                       transform=axs[row, col].transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
