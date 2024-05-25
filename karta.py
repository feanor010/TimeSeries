import pandas as pd

df = pd.read_csv("test.csv")

date_columns = [col for col in df.columns if col.startswith('date_tag')]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

for col in date_columns:
    df[col].fillna(method='ffill', inplace=True)

# Заполнение пропущенных значений для числовых данных
value_columns = [col for col in df.columns if col.startswith('value_tag')]
for col in value_columns:
    df[col] = df[col].interpolate()

# Создание общего временного индекса
common_time_index = pd.date_range(start=df[date_columns].min().min(), end=df[date_columns].max().max(), freq='T')

# Создание нового DataFrame с общим индексом времени
df_synced = pd.DataFrame(index=common_time_index)

# Перенос значений и интерполяция на общий временной индекс
for col in date_columns:
    df_synced[col] = df.set_index(date_columns[0])[col].reindex(common_time_index).interpolate()

for col in value_columns:
    df_synced[col] = df.set_index(date_columns[0])[col].reindex(common_time_index).interpolate()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Нормализация данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_synced[value_columns])

# Подготовка данных для LSTM
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 10  # количество предыдущих временных шагов для использования в прогнозе
X, y = create_dataset(scaled_data, look_back)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, len(value_columns))))
model.add(Dense(len(value_columns)))
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Прогнозирование
predictions = model.predict(X_test)
