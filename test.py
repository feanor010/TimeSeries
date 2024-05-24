import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Функция для загрузки данных из CSV
def load_data(file_path):
    data = pd.read_csv(file_path, index_col='DATE', parse_dates=True)
    return data

# Функция для нормализации данных
def normalize_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

# Функция для подготовки данных для LSTM
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Функция для построения графика
def plot_data(data, predictions=None, title='Time Series Data'):
    lastData = data.values[19000:19600:1]
    plt.figure(figsize=(10, 6))
    plt.plot(lastData, label='Actual Data')
    if predictions is not None:
        plt.plot(range(len(lastData), len(lastData) + len(predictions)), predictions, label='Predictions')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

# Функция для создания и обучения модели LSTM
def train_lstm_model(X_train, y_train, epochs=20, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return model

# Основная функция для загрузки данных, нормализации, обучения модели и прогнозирования
def main(file_path, time_steps=50, epochs=10, batch_size=32, future_steps=1000):
    data = load_data(file_path)
    print("Data loaded successfully.")

    # Нормализация данных
    data_normalized, scaler = normalize_data(data)
    print("Data normalized successfully.")

    # Подготовка данных для LSTM
    X, y = create_dataset(data_normalized, time_steps)
    X_train, X_test = X[:-future_steps], X[-future_steps:]
    y_train, y_test = y[:-future_steps], y[-future_steps:]
    print("Data prepared for LSTM successfully.")

    # Обучение модели LSTM
    model = train_lstm_model(X_train, y_train, epochs, batch_size)
    print("LSTM model trained successfully.")

    # Прогнозирование будущих значений
    predictions = []
    current_batch = X_test[0].reshape((1, time_steps, X_test.shape[2]))
    for i in range(future_steps):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    predictions = scaler.inverse_transform(predictions)
    print("Future data predicted successfully.")
    # Построение графика
    plot_data(data, predictions, title='Time Series Data and Predictions')
    print("Graph plotted successfully.")

# Вызов основной функции
file_path = 'test.csv'  # Укажите путь к вашему CSV файлу
main(file_path)