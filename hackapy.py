import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fedot import Fedot, FedotBuilder
from fedot.core.data.data import InputData, PathType, DataTypesEnum
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from sklearn.metrics import mean_squared_error
from pylab import rcParams


def pre_process():
    def load_data(path):
        return pd.read_csv(path, parse_dates=True)

    def filter_index(data_frame):
        data_frame.drop(columns = ["Unnamed: 0"], inplace = True)

    def break_data(data_frame: pd.DataFrame):
        datas = []

        for column in data_frame:
            prefix, index = str(column).split("_tag_")
            if (prefix == "date"):
                datas.append(pd.DataFrame(columns=["date", "values"]));
                datas[int(index)]["date"] = data_frame[str(column)].dropna()
            else:
                datas[int(index)]["values"] = data_frame[str(column)].dropna()
        return datas

    df = load_data("1_model.csv")
    filter_index(df)

    frames = break_data(df)
    return frames;


compos_conf = {
    "max_depth": 4,
    "max_arity": 3,
    "pop_size": 20,
    "num_of_generations": 4,
    "learning_time": 1,
    "preset": "light_tun"
}

def build_fedot_model(conf, task):
    return(FedotBuilder(problem="ts_forecasting")
       .setup_pipeline_structure(max_depth=conf['max_depth'], max_arity=conf['max_arity'])
       .setup_evolution(pop_size=conf['pop_size'], num_of_generations=conf['num_of_generations'])
       .setup_composition(timeout=conf['learning_time'], preset=conf['preset'], task_params=task.task_params)
       ).build()

def slice_train_test(fed_frame):
    train_data, test_data = train_test_data_setup(fed_frame)
    return train_data, test_data

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

def fedot_process(df: pd.DataFrame):
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=75))
    # times = df[["time"]].set_index("time")
    # values = df[["value"]].set_index("value")
    # print(values)
    # print(df.values())
    # times = df.set_index("time")
    print(df)
    fed_data = InputData.from_dataframe(features_df=df[["date"]], task=task, target_df=df[["values"]])
    model = build_fedot_model(compos_conf, task)
    # print(times)
    train_data, test_data = slice_train_test(fed_data)

    # print(train_data)
    chain = model.fit(train_data)

    # forecast = chain.predict(test_data)
    # forecast_values = forecast.predict
    # plot_forecast(train_data, test_data, forecast_values)
    

frames = pre_process()

fedot_process(frames[0])
