from options_ai_prediction import run_backtest

import os
import time

import numpy as np
import pandas as pd
from keras.initializers import HeNormal
from keras.layers import Dense, Input, Normalization, Layer
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from sklearn.preprocessing import StandardScaler

pd.options.display.max_columns = None

DATA_PATH = 'options-data/selected-1-0-30--100-100-C.csv'
DTE = 1
SEED = 1
TEST_SPLIT = 0.2
BUY_OR_SELL = 'sell'

NEURONS = [512, 256, 128, 64]
ACTIVATION = 'relu'
LEARNING_RATE = 10 ** -4
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

EPOCHS = 100
BATCH_SIZE = 512
LOSS = 'mean_absolute_error'


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[['c_date', 'expiration_date', 'price_strike', 'price', 'underlying_price', 'iv', 'preiv', 'delta', 'gamma',
             'vega', 'theta', 'rho', 'volume', 'open_expiry_price', 'close_expiry_price', 'days_behind',
             'option_symbol']]
    df.dropna(inplace=True)

    df['expiration_distance'] = (pd.to_datetime(df['expiration_date']) - pd.to_datetime(df['c_date'])).dt.days - df[
        'days_behind']
    final_price = df[df['expiration_distance'] == 0]
    final_price.set_index('option_symbol', inplace=True)
    final_price = final_price[~final_price.index.duplicated(keep='first')]
    final_price = final_price['underlying_price']
    df['final_price'] = df['option_symbol'].map(final_price)
    df.dropna(inplace=True)

    df = df[df['expiration_distance'] == 1]
    print(len(df))
    df = df[df['price'] > 0.05]
    print(len(df))

    df['moneyness'] = (df['price_strike'] - df['underlying_price']) / df['underlying_price']

    if BUY_OR_SELL == 'sell':
        df['value'] = (df['price'] - np.maximum(df['final_price'] - df['price_strike'], 0) + df['final_price'] - df[
            'underlying_price']) / df['underlying_price']
    elif BUY_OR_SELL == 'buy':
        df['value'] = (np.maximum(df['final_price'] - df['price_strike'], 0) - df['price']) / df['price']
    else:
        raise ValueError(f'BUY_OR_SELL must be one of "sell" or "buy"')

    rng = np.random.RandomState(SEED)
    shuffle = df['c_date'].unique()
    rng.shuffle(df['c_date'].unique())

    by_date = {}
    for date, row in zip(df['c_date'], df.values):
        by_date.setdefault(date, [])
        by_date[date].append(row)

    split_tracker, train_rows, test_rows = 0, [], []
    for i, date in enumerate(shuffle):
        split_tracker += TEST_SPLIT
        if split_tracker >= 1:
            test_rows.extend(by_date[date])
            split_tracker -= 1
        else:
            train_rows.extend(by_date[date])

    train_df = pd.DataFrame(train_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)

    x_train = train_df.drop(['c_date', 'expiration_date', 'underlying_price', 'price_strike', 'open_expiry_price',
                             'close_expiry_price', 'days_behind', 'value', 'option_symbol', 'final_price',
                             'expiration_distance'], axis=1).values.astype(np.float32)
    x_test = test_df.drop(['c_date', 'expiration_date', 'underlying_price', 'price_strike', 'open_expiry_price',
                           'close_expiry_price', 'days_behind', 'value', 'option_symbol', 'final_price',
                           'expiration_distance'], axis=1).values.astype(np.float32)
    y_train = train_df['value'].values.reshape(-1, 1).astype(np.float32)
    y_test = test_df['value'].values.reshape(-1, 1).astype(np.float32)

    return train_df, test_df, x_train, x_test, y_train, y_test


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


def create_model(x_train):
    model = Sequential()
    model.add(Input(shape=x_train[0].shape))
    norm_layer = Normalization()
    norm_layer.adapt(x_train)
    model.add(norm_layer)

    for n in NEURONS:
        model.add(Dense(n, activation=ACTIVATION, kernel_initializer=HeNormal()))
    model.add(Dense(1))
    # model.add(InverseNormalization(y_train))

    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    return model


def calculate_baseline_loss(y, mean):
    pred = np.full_like(y, mean)
    return np.mean(np.abs(pred - y))


def main():
    start = time.time()

    print(f'({time.time() - start:.2f}s) Loading and processing data...')
    train_df, test_df, x_train, x_test, y_train, y_test = load_data()
    print(f'Data length - Train: {len(train_df)} | Test: {len(test_df)}')

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    print(f'Scaler - Mean: {scaler_y.mean_} | STD: {scaler_y.scale_}')
    test_scaler = StandardScaler()
    test_scaler.fit(y_test)
    print(f'Test Scaler - Mean: {test_scaler.mean_} | STD: {test_scaler.scale_}')
    y_test = scaler_y.transform(y_test)

    print(f'({time.time() - start:.2f}s) Creating model...')
    model = create_model(x_train)

    print(f'({time.time() - start:.2f}s) Calculating baseline metrics...')
    initial_train_loss = model.evaluate(x_train, y_train)
    initial_test_loss = model.evaluate(x_test, y_test)
    baseline_train_loss = calculate_baseline_loss(y_train, mean=scaler_y.mean_)
    baseline_test_loss = calculate_baseline_loss(y_test, mean=scaler_y.mean_)

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
    final_train_loss = model.evaluate(x_train, y_train)
    final_test_loss = model.evaluate(x_test, y_test)
    print(f'Initial and Final Metric Comparison:')
    print(f'Train Loss: {initial_train_loss:.4f} -> {final_train_loss:.4f} | Baseline Train Loss: '
          f'{baseline_train_loss:.4f}')
    print(f'Test Loss: {initial_test_loss:.4f} -> {final_test_loss:.4f} | Baseline Test Loss: {baseline_test_loss:.4f}')

    print(f'({time.time() - start:.2f}s) Saving model...')
    count = 0
    path = f'models/{DTE}-{len(NEURONS)}-{sum(NEURONS)}-{ACTIVATION}-{LEARNING_RATE}-{BATCH_SIZE}-{EPOCHS}-' \
           f'{len(train_df)}-{BUY_OR_SELL}-{count}'
    while os.path.exists(path):
        count += 1
        path = f'models/{DTE}-{len(NEURONS)}-{sum(NEURONS)}-{ACTIVATION}-{LEARNING_RATE}-{BATCH_SIZE}-{EPOCHS}-' \
               f'{len(train_df)}-{BUY_OR_SELL}-{count}'
    model.save(path)

    print(f'({time.time() - start:.2f}s) Running backtest...')
    run_backtest(test_df, x_test, y_test, model, scaler_y)


if __name__ == "__main__":
    main()
