import time
import datetime

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.options.display.max_columns = None
pd.options.display.max_rows = None

DATA_PATH = 'options-data/selected-1-0-30--100-100-C.csv'
DTE = 1
SEED = 1
TEST_SPLIT = 0.2
BUY_OR_SELL = 'sell'

NEURONS = [512, 256, 128, 64]
ACTIVATION = 'relu'
LEARNING_RATE = 10 ** -4

EPOCHS = 100
BATCH_SIZE = 512

NUMBER = 0

TOP_N = 1
MIN_VALUE = 0.005
RISK_TOLERANCE = 0.1

DATE_FORMAT = '%Y-%m-%d'


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


def run_backtest(df, x, y, model, scaler):
    y = y * scaler.scale_ + scaler.mean_
    y_pred = model.predict(x) * scaler.scale_ + scaler.mean_

    by_date = {}
    for date, pred, actual in zip(df['c_date'], y_pred.flatten(), y.flatten()):
        by_date.setdefault(date, {})
        by_date[date][pred] = actual

    roi = 1.0
    returns = []

    for date in sorted(by_date):
        changes = [by_date[date][x] for x in sorted(by_date[date].keys(), reverse=True)[:TOP_N] if x >= MIN_VALUE]
        if not changes:
            continue

        return_ = np.mean(changes)
        if BUY_OR_SELL == 'buy':
            return_ *= RISK_TOLERANCE

        roi *= 1 + return_
        returns.append(return_)

    # For when not daily expirations
    # days = (datetime.datetime.strptime(sorted(by_date)[-1], DATE_FORMAT) - datetime.datetime.strptime(
    #     sorted(by_date)[0], '%Y-%m-%d')).days
    # years = days / 365.25 * TEST_SPLIT

    years = len(by_date) / 252
    annualized_roi = roi ** (1 / years)

    print(f'Years Tested: {years:.1f} | Total ROI: {(roi - 1):,.2%} | Annualized ROI: {annualized_roi - 1:,.2%}')
    print(f'Average Trade Return: {np.mean(returns):,.2%} | Win %: {np.mean([0 if x < 0 else 1 for x in returns]):.2%}')


def main():
    start = time.time()

    print(f'({time.time() - start:.2f}s) Loading and processing data...')
    train_df, test_df, x_train, x_test, y_train, y_test = load_data()

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    print(f'Scaler - Mean: {scaler_y.mean_} | STD: {scaler_y.scale_}')

    print(f'({time.time() - start:.2f}s) Loading model...')
    path = f'models/{DTE}-{len(NEURONS)}-{sum(NEURONS)}-{ACTIVATION}-{LEARNING_RATE}-{BATCH_SIZE}-{EPOCHS}-' \
           f'{len(train_df)}-{BUY_OR_SELL}-{NUMBER}'
    model = keras.models.load_model(path)

    print(f'({time.time() - start:.2f}s) Running backtest...')
    run_backtest(test_df, x_test, y_test, model, scaler_y)


if __name__ == '__main__':
    main()
