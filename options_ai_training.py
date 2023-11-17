import time

import numpy as np
import pandas as pd
from keras.initializers import HeNormal
from keras.layers import Dense, Input, Normalization, Layer
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.options.display.max_columns = None

TEST_SPLIT = 0.2

NEURONS = [512, 256, 128, 64]
ACTIVATION = 'relu'
LEARNING_RATE = 10 ** -3
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

EPOCHS = 20
BATCH_SIZE = 2048
LOSS = 'mean_absolute_error'


def load_data(filepath):
    df = pd.read_parquet(filepath, engine='fastparquet')

    for col in ['strike', 'bid', 'ask']:
        df[col] = df[col] * df['Adjusted close'] / df['Unadjusted close']

    df = df[['date', 'expiration', 'strike', 'ask', 'bid', 'Adjusted close', 'iv', 'preiv', 'delta', 'gamma', 'vega',
             'theta', 'rho', 'volume', 'open interest', 'open_expiry_price', 'close_expiry_price', 'days_behind']]
    df.dropna(inplace=True)

    df['expiration_distance'] = (pd.to_datetime(df['expiration']) - pd.to_datetime(df['date'])).dt.days - df[
        'days_behind']
    df['moneyness'] = (df['strike'] - df['Adjusted close']) / df['Adjusted close']
    df['value'] = (df['bid'] - np.maximum(df['close_expiry_price'] - df['strike'], 0) + df[
        'close_expiry_price'] - df['Adjusted close']) / df['Adjusted close']

    y = df['value'].values.reshape(-1, 1).astype(np.float32)
    df.drop(['date', 'expiration', 'Adjusted close', 'strike', 'open_expiry_price', 'close_expiry_price',
             'days_behind', 'value'], axis=1, inplace=True)
    x = df.values.astype(np.float32)

    return x, y


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


def create_model(neurons, activation, optimizer, loss, x_train, y_train):
    model = Sequential()
    model.add(Input(shape=x_train[0].shape))
    norm_layer = Normalization()
    norm_layer.adapt(x_train)
    model.add(norm_layer)

    for n in neurons:
        model.add(Dense(n, activation=activation, kernel_initializer=HeNormal()))
    model.add(Dense(1))
    # model.add(InverseNormalization(y_train))

    model.compile(optimizer=optimizer, loss=loss)
    return model


def calculate_baseline_loss(y):
    mean_y = np.mean(y, axis=0)
    pred = np.full_like(y, mean_y)
    return np.mean(np.abs(pred - y))


def main():
    start = time.time()

    print(f'({time.time() - start:.2f}s) Loading and processing data...')
    x, y = load_data('venv/options-data/current.parquet')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SPLIT)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    print(scaler_y.mean_, scaler_y.scale_)

    print(f'({time.time() - start:.2f}s) Creating model...')
    model = create_model(NEURONS, ACTIVATION, OPTIMIZER, LOSS, x_train, y_train)

    print(f'({time.time() - start:.2f}s) Calculating baseline metrics...')
    initial_train_loss = model.evaluate(x_train, y_train, verbose=0)
    initial_test_loss = model.evaluate(x_test, y_test, verbose=0)
    baseline_train_loss = calculate_baseline_loss(y_train)
    baseline_test_loss = calculate_baseline_loss(y_test)

    print(f'Train Loss: {initial_train_loss:.4f} | Baseline Train Loss: '
          f'{baseline_train_loss:.4f}')
    print(f'Test Loss: {initial_test_loss:.4f} | Baseline Test Loss: {baseline_test_loss:.4f}')

    print(f'({time.time() - start:.2f}s) Starting training...')
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
    )

    print(f'({time.time() - start:.2f}s) Evaluating model...')
    final_train_loss = model.evaluate(x_train, y_train, verbose=0)
    final_test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f'Initial and Final Metric Comparison:')
    print(f'Train Loss: {initial_train_loss:.4f} -> {final_train_loss:.4f} | Baseline Train Loss: '
          f'{baseline_train_loss:.4f}')
    print(f'Test Loss: {initial_test_loss:.4f} -> {final_test_loss:.4f} | Baseline Test Loss: {baseline_test_loss:.4f}')

    print(f'({time.time() - start:.2f}s) Saving model...')
    model.save(f'models/{len(NEURONS)}-{max(NEURONS)}-{LEARNING_RATE}-{EPOCHS}-{BATCH_SIZE}-{int(start)}')


if __name__ == "__main__":
    main()
