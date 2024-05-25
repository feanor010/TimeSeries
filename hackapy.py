import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fedot import Fedot, FedotBuilder
from fedot.core.data.data import InputData, PathType, DataTypesEnum
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

from sklearn.metrics import mean_squared_error
from pylab import rcParams


def load_data(path):
    return pd.read_csv(path, delimiter=",")


def filter_index(data_frame):
    data_frame.drop(columns=["Unnamed: 0"], inplace=True)


def break_data(data_frame: pd.DataFrame):
    datas = []

    for column in data_frame:
        prefix, index = str(column).split("_tag_")
        if prefix == "date":
            datas.append(pd.DataFrame(columns=["date", "values"]))

            to_pip = data_frame[[str(column)]]

            to_pip = to_pip.dropna()
            to_pip[str(column)] = pd.to_datetime(to_pip[str(column)], format="mixed")


            # to_pip[str(column)] = pd.to_datetime(to_pip[str(column)])
            datas[int(index)]["date"] = to_pip[str(column)]
        else:
            datas[int(index)]["values"] = data_frame[str(column)].dropna()

    return datas


def pre_process():
    df = load_data("2_model.csv")
    filter_index(df)

    frames = break_data(df)
    return frames;


compos_conf = {
    "max_depth": 8,
    "max_arity": 4,
    "pop_size": 50,
    "num_of_generations": 60,
    "learning_time": 4,
    "preset": "best-quality"
}


def build_fedot_model(conf, task):
    return (FedotBuilder(problem="ts_forecasting")
            .setup_pipeline_structure(max_depth=conf['max_depth'], max_arity=conf['max_arity'])
            .setup_evolution(pop_size=conf['pop_size'], num_of_generations=conf['num_of_generations'])
            .setup_composition(timeout=conf['learning_time'], preset=conf['preset'], task_params=task.task_params)
            ).setup_parallelization(-1).build()


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
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=100))
    # times = df[["time"]].set_index("time")
    # values = df[["value"]].set_index("value")
    # print(values)
    # print(df.values())
    # times = df.set_index("time")

    # fed_data = InputData.from_csv_time_series("2_model.csv", task=task, delimiter=",", target_column="value_tag_0",
    #                                           index_col="date_tag_0")
    # df["date"] = pd.to_datetime(df["date"])
    # df["date"] = pd.to_datetime(df["date"])
    # print(df.values)
    features = df["date"]
    print(features.to_numpy())
    target = df["values"]
    print(target.to_numpy())
    fed_data = InputData(features=target.to_numpy(), target=target.to_numpy(), task=task, idx=features.to_numpy(), data_type=DataTypesEnum.ts)

    # fed_data = InputData.from_dataframe(features_df=df[["DATE"]], task=task, target_df=df[["value"]])

    # print(fed_data)
    # print(fed_data.features)
    # print(df[["date"]].set_index("date"))
    # print(fed_data)
    model = build_fedot_model(compos_conf, task)
    # print(times)
    train_data, test_data = slice_train_test(fed_data)

    # print(train_data)
    chain = model.fit(train_data)

    forecast = chain.predict(test_data)
    forecast_values = forecast.predict
    plot_forecast(train_data, test_data, forecast_values)


frames = pre_process()

fedot_process(frames[0])

# fedot_process(load_data("test.csv"))
