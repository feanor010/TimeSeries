import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

conf = {
    "seq_len": 10,
    "valid_splt": 0.01
}

class TimeSeriesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Predictor")
        self.epochs_label = tk.Label(root, text="Epochs:")
        self.epochs_label.pack()
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.insert(tk.END, "5") 
        self.epochs_entry.pack()
        self.filename = None
        self.df = None
        self.scaler = MinMaxScaler()
        self.date_columns = []
        self.value_columns = []
        self.split_ratio_label = None
        self.select_file_button = tk.Button(root, text="Select File", command=self.load_file)
        self.select_file_button.pack()

        self.split_ratio_slider = ttk.Scale(root, from_=0.8, to=0.99, orient='horizontal', command=self.update_split_ratio)
        self.split_ratio_slider.set(0.9)
        self.split_ratio_slider.pack()
        self.split_ratio_label = tk.Label(root, text="Split Ratio: 0.9")
        self.split_ratio_label.pack()

        self.train_button = tk.Button(root, text="Train and Predict", command=self.train_and_predict)
        self.train_button.pack()

        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack()

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filename:
            self.df = pd.read_csv(self.filename)
            self.preprocess_data()

    def preprocess_data(self):
        if self.df is not None:
            self.date_columns = [col for col in self.df.columns if 'date_tag' in col]
            self.value_columns = [col for col in self.df.columns if 'value_tag' in col]

            for col in self.date_columns:
                self.df[col] = pd.to_datetime(self.df[col])

            self.df.dropna(subset=self.date_columns + self.value_columns, inplace=True)
            self.df.interpolate(method='ffill', inplace=True)
            self.df[self.value_columns] = self.df[self.value_columns].apply(pd.to_numeric, errors='coerce')
            self.df.dropna(inplace=True)

            self.df[self.value_columns] = self.scaler.fit_transform(self.df[self.value_columns])

    def update_split_ratio(self, val):
        self.split_ratio_label.config(text=f"Split Ratio: {float(val):.2f}")

    def train_and_predict(self):
        if self.df is None:
            return

        split_ratio = self.split_ratio_slider.get()
        seq_length = conf['seq_len']
        X, y = [], []

        for i in range(len(self.df) - seq_length):
            X.append(self.df[self.value_columns].iloc[i:i+seq_length].values)
            y.append(self.df[self.value_columns].iloc[i+seq_length].values)

        X = np.array(X)
        y = np.array(y)

        split = int(split_ratio * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        epochs = int(self.epochs_entry.get()) 

        model = Sequential([
            LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(256, return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dense(y_train.shape[1])
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=conf['valid_splt']) 
        y_pred = model.predict(X_test)

        y_train_denorm = self.scaler.inverse_transform(y_train)
        y_test_denorm = self.scaler.inverse_transform(y_test)
        y_pred_denorm = self.scaler.inverse_transform(y_pred)

        metrics = {}
        for i, col in enumerate(self.value_columns):
            mae = mean_absolute_error(y_test_denorm[:, i], y_pred_denorm[:, i])
            mse = mean_squared_error(y_test_denorm[:, i], y_pred_denorm[:, i])
            metrics[col] = {'MAE': mae, 'MSE': mse}
            print(f'{col}: MAE = {mae:.4f}, MSE = {mse:.4f}')

        self.plot_results(y_train_denorm, y_test_denorm, y_pred_denorm, metrics)

    def plot_results(self, y_train_denorm, y_test_denorm, y_pred_denorm, metrics):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        num_plots = len(self.value_columns)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7*num_rows))
        concat_array = np.vstack([y_train_denorm, y_test_denorm])
        
        if num_plots == 1:
            axs = [axs]

        real_indices = np.arange(len(y_test_denorm) + len(y_train_denorm))
        pred_indices = np.arange(len(y_test_denorm) - len(y_pred_denorm), len(y_test_denorm))
        real_dates = self.df[self.date_columns[0]].iloc[real_indices].values
        pred_dates = self.df[self.date_columns[0]].iloc[int(self.split_ratio_slider.get() * len(self.df)) + pred_indices].values

        for i, ax in enumerate(axs.flat):
            if i >= num_plots:
                ax.axis('off')
                continue
            ax.plot(real_dates, concat_array[:, i], label=f'Real {self.value_columns[i]}', color='blue', linestyle='--')
            ax.plot(pred_dates, y_pred_denorm[:, i], label=f'Predicted {self.value_columns[i]}', color='red', linestyle ='-')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(f'Comparison of {self.value_columns[i]} over the whole test period')
            ax.text(1, 1, f"MAE: {metrics[self.value_columns[i]]['MAE']:.4f}\nMSE: {metrics[self.value_columns[i]]['MSE']:.4f}",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.legend()

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesPredictorApp(root)
    root.mainloop()
