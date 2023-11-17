import datetime
import time

import keras
import numpy as np
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.max_rows = 10_000


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

    df = df[df['expiration_distance'] == 1]
    df.sort_values(by=['date'], ascending=True, inplace=True)

    y = df['value'].values.reshape(-1, 1).astype(np.float32)
    pred_df = df.drop(['date', 'expiration', 'Adjusted close', 'strike', 'open_expiry_price', 'close_expiry_price',
                       'days_behind', 'value'], axis=1)
    x = pred_df.values.astype(np.float32)

    return df, x, y


def run_backtest(df, x, y, model, mean, std):
    y_pred = model.predict(x) * std + mean

    by_date = {}
    for date, pred, actual in zip(df['date'], y_pred.flatten(), y.flatten()):
        print(date)
        by_date.setdefault(date, {})
        by_date[date][pred] = actual

    roi = 1.0
    for date in sorted(by_date):

        max_pred = max(by_date[date].keys())
        if max_pred < 0:
            continue
        change = by_date[date][max_pred]
        roi *= (1 + change)

    years_tested = (datetime.datetime.strptime(df['date'].iloc[len(df) - 1], '%Y-%m-%d') - datetime.datetime.strptime(
        df['date'].iloc[0], '%Y-%m-%d')).days / 365.25
    annualized_roi = roi ** (1 / years_tested)

    print(f'Years Tested: {years_tested:.1f} | Total ROI: {(roi - 1):,.2%} | Annualized ROI: {annualized_roi - 1:,.2%}')


def main():
    start = time.time()
    print(f'({time.time() - start:.2f}s) Loading and processing data...')
    df, x, y = load_data('venv/options-data/100-1700017473.parquet')
    print(f'({time.time() - start:.2f}s) Loading model...')
    model = keras.models.load_model('venv/models/4-512-0.001-20-2048-1700152318')
    print(f'({time.time() - start:.2f}s) Running backtest...')
    run_backtest(df, x, y, model, mean=-0.00764923, std=0.04477473)


if __name__ == '__main__':
    main()
