import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
df = pd.read_csv("1_model.csv")

# Преобразование дат
date_columns = [col for col in df.columns if 'date_tag' in col]
value_columns = [col for col in df.columns if 'value_tag' in col]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# Удаление строк с пропущенными значениями в столбцах с датами и значениями
df.dropna(subset=date_columns + value_columns, inplace=True)

# Создание DataFrame для временных рядов
data = pd.DataFrame()

# Добавление столбцов с датами в качестве индексов
for i, col in enumerate(date_columns):
    data[col] = df[col]
    data[f'Value_{i}'] = df[value_columns[i]]

# Установка столбца с датами как индекс
data.set_index(date_columns[0], inplace=True)

# Интерполяция данных для заполнения пропущенных значений
data.interpolate(method='ffill', inplace=True)

# Разделение данных на числовые и даты
numeric_data = data[value_columns]
date_data = data[date_columns]
# Нормализация числовых данных
scaler = MinMaxScaler()
numeric_data_normalized = scaler.fit_transform(numeric_data)

# Соединение числовых данных с датами
data_normalized = pd.DataFrame(numeric_data_normalized, columns=numeric_data.columns)
data_normalized[date_columns] = date_data

# Функция для создания последовательностей временных рядов
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Длина последовательности для обучения модели
seq_length = 10

# Создание последовательностей
X, y = create_sequences(data_normalized, seq_length)

# Разделение данных на обучающий и тестовый наборы
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Создание модели RNN
model = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    SimpleRNN(64),
    Dense(y_train.shape[1])
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse')

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Прогнозирование
y_pred = model.predict(X_test)

# Восстановление оригинальных масштабов данных
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Построение графиков
plt.figure(figsize=(14, 7))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', color='darkgreen')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('RNN Prediction')
plt.legend()
plt.show()
