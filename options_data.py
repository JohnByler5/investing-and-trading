import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

import ivolatility as ivol

pd.options.display.max_columns = None

DATE_FORMAT = '%Y-%m-%d'

SYMBOLS_FILE_PATH = 'tickers/s&p500.csv'
NAME_EXTRA = '-with-hv'

START_DATE = datetime.datetime.strptime('2001-11-01', DATE_FORMAT)
END_DATE = datetime.datetime.today()
LOAD_PREV_DF = True
EXTRA_COLS = ['unadjusted_open', 'unadjusted_close', '10d HV', '20d HV', '30d HV', '60d HV', '90d HV', '120d HV',
              '150d HV', '180d HV', 'days_behind']

DTE_FROM = 0
DTE_TO = 30
MONEYNESS_FROM = -100
MONEYNESS_TO = 100
OPTION_TYPE = 'C'

MAX_THREADS = 20
TIMEOUT = 5.0
PAUSE = 0.0

LOCK = threading.Lock()

ivol.set_login_params(api_key='S3kG46572xj4yMg4')
get_options_data = ivol.set_method('/equities/eod/stock-opts-by-param')
get_price_data = ivol.set_method('/equities/eod/stock-prices')
get_hv_data = ivol.set_method('/equities/eod/hv')


class OptionsDataCollector:
    def __init__(self, symbols, name):
        self.save_path = f'options-data/{name}-{len(symbols)}-{DTE_FROM}-{DTE_TO}-' \
                         f'{MONEYNESS_FROM}-{MONEYNESS_TO}-{OPTION_TYPE}.parquet'
        if LOAD_PREV_DF and os.path.exists(self.save_path):
            self.df = pd.read_parquet(self.save_path)
        else:
            self.df = pd.DataFrame()
        self.to_concat = [self.df]

        symbols_done = set() if self.df.empty else set(self.df['symbol'])
        symbols_to_do = [symbol for symbol in symbols['Symbol'] if symbol not in symbols_done]
        self.symbols = symbols_to_do

        # nyse_dates = xcal.get_calendar(
        #     'NASDAQ', start=START_DATE.strftime(DATE_FORMAT), end=END_DATE.strftime(DATE_FORMAT)).schedule['open']
        # nasdaq_dates = xcal.get_calendar(
        #     'NASDAQ', start=START_DATE.strftime(DATE_FORMAT), end=END_DATE.strftime(DATE_FORMAT)).schedule['open']
        # self.dates = list(sorted(set(list(nyse_dates) + list(nasdaq_dates))))

        self.futures = set()
        self.executor = None

        self.start = None

    def run(self):
        self.start = time.time()
        self.update('Starting...')

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as self.executor:
            for i, symbol in enumerate(self.symbols, 1):
                self.update(f'({i:,} / {len(self.symbols):,}) {symbol}')
                self.process_symbol(symbol)

    def process_symbol(self, symbol):
        price_data = self.get_price_data(symbol)
        if price_data.empty:
            self.update(f'Skipping {symbol} because of empty price data', red=True)
            return

        for result in self.process_dates(symbol, price_data.index):
            if result.empty:
                continue
            self.to_concat.append(self.fix_data(self.add_data(result, price_data), symbol))

        self.save_data()

    def process_dates(self, symbol, dates):
        self.update(f'Gathering options data for {len(dates):,} dates...')

        for i, date in enumerate(dates, 1):
            if isinstance(date, datetime.datetime):
                date = date.strftime('%Y-%m-%d')
            self.futures.add(self.executor.submit(self.get_options_data, symbol, date))

            for result in self.manage_futures():
                yield result

            if i % 100 != 0:
                continue
            self.update(f'    ({i:,} / {len(dates):,}) dates processed')

        self.update(f'Finished processing {len(dates):,} dates')

    def get_options_data(self, symbol, date):
        error, results = get_options_data(symbol=symbol, tradeDate=date, dteFrom=DTE_FROM, dteTo=DTE_TO,
                                          moneynessFrom=MONEYNESS_FROM, moneynessTo=MONEYNESS_TO, cp=OPTION_TYPE,
                                          timeout=TIMEOUT, pause=PAUSE)

        if error is not None:
            self.update(f'Error: {error} - {symbol} {date}', red=True)
        elif results.empty:
            self.update(f'Empty Results - {symbol} {date}', red=True)
        elif 'error' in results.columns:
            results = pd.DataFrame()
            self.update(f'Error: {results["error"].iloc[0]} - {symbol} {date}', red=True)

        return results

    @staticmethod
    def create_new_row(date, price_data):
        if date is None:
            return [None for _ in range(len(EXTRA_COLS))]

        attempts = 0
        while attempts < 5:
            try:
                return [attempts if col == 'days_behind' else price_data[col][date] for col in EXTRA_COLS]
            except KeyError:
                date = (datetime.datetime.strptime(date, DATE_FORMAT) - datetime.timedelta(days=1)).strftime(
                    DATE_FORMAT)
                attempts += 1

        return [None for _ in range(len(EXTRA_COLS))]

    def add_data(self, df, price_data):
        if df.empty:
            return df

        df['c_date'] = df['c_date'].str.split().str[0]
        df['expiration_date'] = df['expiration_date'].str.split().str[0]

        new_data = []
        for date in df['expiration_date'].values:
            new_data.append(self.create_new_row(date, price_data))

        df[EXTRA_COLS] = new_data
        return df

    @staticmethod
    def fix_data(df, symbol):
        df = df.dropna(axis=0, how='all')
        if df.empty:
            return df
        df['symbol'] = symbol
        df['price_strike'] = df['price_strike'].astype(np.float32)
        df['option_symbol'] = df['symbol'].str.ljust(6) + pd.to_datetime(
            df['expiration_date'], format=DATE_FORMAT).dt.strftime('Ymd') + df['call_put'] + df[
            'price_strike'].astype(str).str.split('.').str[0].str.rjust(5) + df[
            'price_strike'].astype(str).str.split('.').str[1].str.ljust(3)
        return df

    def get_data(self, func, context, *args, **kwargs):
        attempts, error, data = 0, None, pd.DataFrame()

        while attempts < 3:
            error, data = func(*args, **kwargs)
            if error is not None or data.empty or 'error' in data.columns:
                attempts += 1
                continue
            return data

        if error is not None:
            self.update(f'Error: {error} - {context}', red=True)
        elif data.empty:
            self.update(f'Empty Results - {context}', red=True)
        elif 'error' in data.columns:
            results = pd.DataFrame()
            self.update(f'Error: {results["error"].iloc[0]} - {context}', red=True)
        else:
            self.update('How did we get here? Something went wrong.', red=True)

        return data

    def get_price_data(self, symbol):
        price_data = self.get_data(get_price_data, context=symbol, symbol=symbol,
                                   from_=START_DATE.strftime(DATE_FORMAT), to=END_DATE.strftime(DATE_FORMAT),
                                   timeout=TIMEOUT, pause=PAUSE)
        if price_data.empty:
            return price_data

        price_data['unadjusted_open'] = price_data['open'] * price_data['unadjusted_close'] / price_data['close']
        price_data['date'] = price_data['date'].str.split().str[0]
        price_data.set_index('date', drop=True, inplace=True)

        hv_data = self.get_data(get_hv_data, context=symbol, symbol=symbol, from_=price_data.index[-1],
                                to=price_data.index[0], timeout=TIMEOUT, pause=PAUSE)
        if hv_data.empty:
            return hv_data

        hv_data['date'] = hv_data['date'].str.split().str[0]
        hv_data.set_index('date', drop=True, inplace=True)
        hv_data = hv_data[hv_data.index.isin(price_data.index)]
        price_data = pd.concat([price_data, hv_data], axis=1)

        return price_data

    def manage_futures(self):
        if len(self.futures) < MAX_THREADS * 10:
            return

        for future in as_completed(self.futures):
            self.futures.remove(future)

            if future.exception():
                self.update(f'Exception in thread: {future.exception()}', red=True)
            else:
                yield future.result()

            if len(self.futures) <= MAX_THREADS * 5:
                break

    def save_data(self):
        self.df = pd.concat(self.to_concat, ignore_index=True)
        self.to_concat = [self.df]
        self.update(f'Saving {self.df.memory_usage(deep=True).sum() / 1024 ** 3:,.2f}GB of data...')
        self.df.to_parquet(self.save_path)

    def update(self, message, red=False):
        if red:
            message = f'\033[31m{message}\033[0m'

        with LOCK:
            print(f'({datetime.timedelta(seconds=int(round(time.time() - self.start)))}) {message}')


def main():
    name = SYMBOLS_FILE_PATH.split('/')[1].replace('.csv', '') + NAME_EXTRA
    collector = OptionsDataCollector(pd.read_csv(SYMBOLS_FILE_PATH), name)
    collector.run()


if __name__ == '__main__':
    main()
