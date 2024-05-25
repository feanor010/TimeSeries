import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fedot import Fedot, FedotBuilder
from fedot.core.data.data import InputData, PathType
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from sklearn.metrics import mean_squared_error
from pylab import rcParams

rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')

# Настройка задачи и параметров
task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=75))

# Загрузка данных
data = InputData.from_csv_time_series("test.csv", task=task, delimiter=",", target_column="value", index_col="DATE")

# Разделение данных на обучающую и тестовую выборки
train_data, test_data = train_test_data_setup(data)
print(train_data)

# Параметры композиции
com_par = {
    "max_depth": 4,
    "max_arity": 3,
    "pop_size": 20,
    "num_of_generations": 4,
    "learning_time": 1,
    "preset": "light_tun"
}

# Создание и обучение модели FEDOT
fed = (FedotBuilder(problem="ts_forecasting")
       .setup_pipeline_structure(max_depth=com_par['max_depth'], max_arity=com_par['max_arity'])
       .setup_evolution(pop_size=com_par['pop_size'], num_of_generations=com_par['num_of_generations'])
       .setup_composition(timeout=com_par['learning_time'], preset=com_par['preset'], task_params=task.task_params)
       ).build()

chain = fed.fit(train_data)

forecast = chain.predict(test_data)
forecast_values = forecast.predict

def plot_forecast(train_data, test_data, forecast_data):
    plt.figure(figsize=(18, 7))

    plt.plot(train_data.idx, train_data.target, label='Train data', color='blue')

    plt.plot(test_data.idx, test_data.target, label='Test data', color='green')

    forecast_index = test_data.idx
    plt.plot(forecast_index, forecast_data, label='Forecast', color='red')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_forecast(train_data, test_data, forecast_values)
