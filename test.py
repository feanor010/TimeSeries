import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
df = pd.read_csv("1_model.csv")

# Идентификация столбцов с датами и значениями
date_columns = [col for col in df.columns if 'date_tag' in col]
value_columns = [col for col in df.columns if 'value_tag' in col]

# Преобразование дат в формат datetime
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# Удаление строк с пропущенными значениями в столбцах с датами и значениями
df.dropna(subset=date_columns + value_columns, inplace=True)

# Интерполяция пропущенных значений
df.interpolate(method='ffill', inplace=True)

# Преобразование значений в числовой формат
df[value_columns] = df[value_columns].apply(pd.to_numeric, errors='coerce')

# Удаление строк с пропущенными значениями после преобразований
df.dropna(inplace=True)

# Нормализация данных
scaler = MinMaxScaler()
df[value_columns] = scaler.fit_transform(df[value_columns])

# Определение длины последовательности
seq_length = 100

# Создание последовательностей
X = []
y = []

for i in range(len(df) - seq_length):
    X.append(df[value_columns].iloc[i:i+seq_length].values)
    y.append(df[value_columns].iloc[i+seq_length].values)

X = np.array(X)
y = np.array(y)

# Разделение данных на обучающий и тестовый наборы
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Создание модели LSTM
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(y_train.shape[1])
])

# Компиляция модели
model.compile(optimizer=RMSprop(), loss='mse')

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Предсказание для последних 10 дней
X_pred = X_test[-10:]
y_pred = model.predict(X_pred)

# Создание subplot
num_plots = len(value_columns)
num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7*num_rows))

# Построение графиков сравнения предсказанных и реальных значений для последних 10 дней
for i in range(num_plots):
    row = i // num_cols
    col = i % num_cols

    # Индексы реальных и предсказанных значений
    real_indices = np.arange(len(y_test) - 10, len(y_test))
    pred_indices = np.arange(len(y_test), len(y_test) + 10)

    # Даты для реальных и предсказанных значений
    real_dates = df[date_columns[0]].iloc[split+len(y_test)-10:split+len(y_test)].values
    pred_dates = df[date_columns[0]].iloc[split+len(y_test):split+len(y_test)+10].values

    axs[row, col].plot(real_dates, y_test[-10:, i], label=f'Реальные {value_columns[i]}', color='blue', linestyle='-')
    axs[row, col].plot(pred_dates, y_pred[:, i], label=f'Предсказанные {value_columns[i]}', color='green', linestyle='--')
    axs[row, col].set_xlabel('Время')
    axs[row, col].set_ylabel('Значение')
    axs[row, col].set_title(f'Сравнение {value_columns[i]} за последние 10 дней')
    axs[row, col].legend()

plt.tight_layout()
plt.show()

# Визуализация графика обучения и валидации
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss (обучение)')
plt.plot(history.history['val_loss'], label='Loss (валидация)')
plt.title('График обучения и валидации')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.show()