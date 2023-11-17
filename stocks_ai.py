import time

import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.initializers import HeNormal
from keras.layers import Dense, Input, Normalization, Layer
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_DAYS = 5
FEATURES = ['Adj Close % Change', 'Open % Change', 'High % Change', 'Low % Change', 'Volume % Change']
TARGET = 'Adj Close % Change'
TEST_SPLIT = 0.2

NEURONS = [512, 256, 128, 64, 32, 16, 8, 4, 2]
ACTIVATION = 'relu'
LEARNING_RATE = 10 ** -5
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

EPOCHS = 100
BATCH_SIZE = 32
LOSS = 'mean_absolute_error'


class AnnualizedROICallback(Callback):
    def __init__(self, x_test, y_test, scaler_y):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.scaler_y = scaler_y

    def on_epoch_end(self, epoch, logs=None):
        model_roi, baseline_roi, algorithm_roi = test_model(
            self.model, self.x_test, self.y_test, self.scaler_y)
        print(f' - Model ROI: {model_roi:.2%} | Baseline ROI: {baseline_roi:.2%} | Algorithm ROI: {algorithm_roi:.2%}')


class InverseNormalization(Layer):
    def __init__(self, data, **kwargs):
        super(InverseNormalization, self).__init__(**kwargs)
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def call(self, inputs, training=None, **kwargs):
        if training:
            return inputs
        else:
            return inputs * self.scaler.scale_ + self.scaler.mean_

    def compute_output_shape(self, input_shape):
        return input_shape


def load_and_process_data(filepath):
    df = pd.read_csv(filepath)
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            continue
        df[column + ' % Change'] = df[column].pct_change()
    df.dropna(inplace=True)
    return df


def prepare_data(df, window_length, feature_labels, target_label):
    features = []

    for i in reversed(range(window_length)):
        section = df.iloc[i:len(df) - window_length + i]
        for feature_label in feature_labels:
            features.append(section[feature_label].values)

    x = np.array(features).T
    y = df[target_label].values[window_length:]
    return x, y.reshape(-1, 1)


def create_model(neurons, activation, optimizer, loss, x_train, y_train):
    model = Sequential()
    model.add(Input(shape=x_train[0].shape))
    norm_layer = Normalization()
    norm_layer.adapt(x_train)
    model.add(norm_layer)

    for n in neurons:
        model.add(Dense(n, activation=activation, kernel_initializer=HeNormal()))
    model.add(Dense(1))
    model.add(InverseNormalization(y_train))

    model.compile(optimizer=optimizer, loss=loss)
    return model


def test_model(model, x_test, y_test, scaler_y):
    y_test = scaler_y.inverse_transform(y_test).flatten()
    predictions = model.predict(x_test).flatten()

    roi = 1.0
    for pred, y in zip(predictions, y_test):
        if pred > 0:
            roi *= (1 + y)
    model_roi = roi ** (252 / len(x_test)) - 1

    roi = 1.0
    for x, y in zip(x_test, y_test):
        roi *= (1 + y)
    baseline_roi = roi ** (252 / len(x_test)) - 1

    roi = 1.0
    for x, y in zip(x_test, y_test):
        if x[0] < 0.011:
            roi *= (1 + y)
    algorithm_roi = roi ** (252 / len(x_test)) - 1

    return model_roi, baseline_roi, algorithm_roi


def calculate_baseline_loss(y):
    mean_y = np.mean(y, axis=0)
    pred = np.full_like(y, mean_y)
    return np.mean(np.abs(pred - y))


def calculate_algorithm_loss(x_train, y_train, x_test, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)
    return np.mean(np.abs(y_pred - y_test))


def main():
    start = time.time()

    print(f'({time.time() - start:.2f}s) Loading and processing data...')
    df = load_and_process_data('venv/stock-data/SPY_1993-01-29_2023-11-10.csv')
    x, y = prepare_data(df, window_length=INPUT_DAYS, feature_labels=FEATURES, target_label=TARGET)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SPLIT)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    print(f'({time.time() - start:.2f}s) Creating model...')
    model = create_model(NEURONS, ACTIVATION, OPTIMIZER, LOSS, x_train, y_train)

    print(f'({time.time() - start:.2f}s) Calculating baseline metrics...')
    initial_train_loss = model.evaluate(x_train, y_train, verbose=0)
    initial_test_loss = model.evaluate(x_test, y_test, verbose=0)
    baseline_train_loss = calculate_baseline_loss(y_train)
    baseline_test_loss = calculate_baseline_loss(y_test)
    algorithm_train_loss = calculate_algorithm_loss(x_train, y_train, x_train, y_train)
    algorithm_test_loss = calculate_algorithm_loss(x_train, y_train, x_test, y_test)

    initial_model_roi, baseline_roi, algorithm_roi = test_model(model, x_test, y_test, scaler_y)
    print(f'Initial Train Loss: {initial_train_loss:.4f} | Baseline Train Loss: {baseline_train_loss:.4f} | Algorithm '
          f'Train Loss: {algorithm_train_loss:.4f}')
    print(f'Initial Test Loss: {initial_test_loss:.4f} | Baseline Test Loss: {baseline_test_loss:.4f} | Algorithm '
          f'Test Loss: {algorithm_test_loss:.4f}')
    print(f'Initial Model ROI: {initial_model_roi:.2%} | Baseline ROI: {baseline_roi:.2%} | Algorithm ROI: '
          f'{algorithm_roi:.2%}')

    print(f'({time.time() - start:.2f}s) Starting training...')
    roi_callback = AnnualizedROICallback(x_test, y_test, scaler_y)
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[roi_callback]
    )

    print(f'({time.time() - start:.2f}s) Evaluating model...')
    final_train_loss = model.evaluate(x_train, y_train, verbose=0)
    final_test_loss = model.evaluate(x_test, y_test, verbose=0)
    final_model_roi, baseline_roi, algorithm_roi = test_model(model, x_test, y_test, scaler_y)
    print(f'Final Train Loss: {final_train_loss:.4f} | Baseline Train Loss: {baseline_train_loss:.4f}')
    print(f'Final Test Loss: {final_test_loss:.4f} | Baseline Test Loss: {baseline_test_loss:.4f}')
    print(f'Final Model ROI: {final_model_roi:.2%} | Baseline ROI: {baseline_roi:.2%} | Algorithm ROI: '
          f'{algorithm_roi:.2%}')

    print(f'\nComparison of Initial and Final Metrics:')
    print(f'Train Loss: {initial_train_loss:.4f} -> {final_train_loss:.4f} | Baseline Train Loss: '
          f'{baseline_train_loss:.4f} | Algorithm Train Loss: {algorithm_train_loss:.4f}')
    print(f'Test Loss: {initial_test_loss:.4f} -> {final_test_loss:.4f} | Baseline Test Loss: {baseline_test_loss:.4f}'
          f' | Algorithm Test Loss: {algorithm_test_loss:.4f}')
    print(f'Model ROI: {initial_model_roi:.2%} -> {final_model_roi:.2%} | Baseline ROI: {baseline_roi:.2%} | Algorithm '
          f'ROI: {algorithm_roi:.2%}')


if __name__ == "__main__":
    main()
