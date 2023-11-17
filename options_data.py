import datetime
import time
import threading
import exchange_calendars as xcal
import ivolatility as ivol
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

pd.options.display.max_columns = None
DATE_FORMAT = '%Y-%m-%d'
SYMBOLS_FILE_PATH = 'tickers/s&p-500.csv'
START_DATE = datetime.datetime.strptime('2001-11-01', DATE_FORMAT)
END_DATE = datetime.datetime.today()

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


class OptionsDataCollector:
    def __init__(self, symbols, prev_df=None):
        symbols_done = set(prev_df['symbol']) if prev_df is not None and not prev_df.empty else set()
        symbols_to_do = [symbol for symbol in symbols['Symbol'] if symbol not in symbols_done]
        self.symbols = symbols_to_do

        nyse_dates = xcal.get_calendar(
            'NASDAQ', start=START_DATE.strftime(DATE_FORMAT), end=END_DATE.strftime(DATE_FORMAT)).schedule['open']
        nasdaq_dates = xcal.get_calendar(
            'NASDAQ', start=START_DATE.strftime(DATE_FORMAT), end=END_DATE.strftime(DATE_FORMAT)).schedule['open']
        self.dates = list(sorted(set(list(nyse_dates) + list(nasdaq_dates))))

        self.prev_df = pd.DataFrame if prev_df is None else prev_df

        self.futures = set()
        self.data = []

        self.executor = None
        self.start = None

    def run(self):
        self.start = time.time()
        self.update('Starting...')

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as self.executor:
            for i, symbol in enumerate(self.symbols, 1):
                self.update(f'({i:,} / {len(self.symbols):,}) {symbol}')
                self.process_symbol(symbol)

                self.update(f'Saving {sys.getsizeof(self.data) / 1024 ** 2:,.2f}MB of data...')
                self.save_data(f'options-data/{len(self.symbols)}-{int(self.start)}.csv')

    def process_symbol(self, symbol):
        price_data = self.get_price_data(symbol)
        if price_data.empty:
            self.update(f'Skipping {symbol} because of empty price data', red=True)
            return

        self.update(f'Gathering options data for {len(self.dates):,} dates...')
        for result in self.process_dates(symbol):
            self.data.append(self.add_data(result, price_data))

        self.update(f'Finished processing {len(self.dates):,}) dates')

    def process_dates(self, symbol):
        for j, date in enumerate(reversed(self.dates), 1):
            self.futures.add(self.executor.submit(self.get_options_data, symbol, date.strftime('%Y-%m-%d')))

            for result in self.manage_futures():
                yield result

            if j % 100 != 0:
                continue

            self.update(f'    ({j:,} / {len(self.dates):,}) dates processed')

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
            return [None, None, None]

        attempts = 0
        while attempts < 5:
            try:
                return [price_data['open'][date], price_data['close'][date], attempts]
            except KeyError:
                date = (datetime.datetime.strptime(date, DATE_FORMAT) - datetime.timedelta(days=1)).strftime(
                    DATE_FORMAT)
                attempts += 1

        return [None, None, None]

    def add_data(self, df, price_data):
        if df.empty:
            return df

        df['c_date'] = df['c_date'].str.split().str[0]
        df['expiration_date'] = df['expiration_date'].str.split().str[0]

        new_data = []
        for date in df['expiration_date'].values:
            new_data.append(self.create_new_row(date, price_data))

        df[['open_expiry_price', 'close_expiry_price', 'days_behind']] = new_data
        return df

    def get_price_data(self, symbol):
        attempts, error, price_data = 0, None, pd.DataFrame()
        while attempts < 3:
            error, price_data = get_price_data(symbol=symbol, from_=START_DATE.strftime(DATE_FORMAT),
                                               to=END_DATE.strftime(DATE_FORMAT), timeout=TIMEOUT, pause=PAUSE)
            if error is not None or price_data.empty or 'error' in price_data.columns:
                attempts += 1
                continue
            break

        else:
            if error is not None:
                self.update(f'Error: {error} - {symbol}', red=True)
            elif price_data.empty:
                self.update(f'Empty Results - {symbol}', red=True)
            elif 'error' in price_data.columns:
                results = pd.DataFrame()
                self.update(f'Error: {results["error"].iloc[0]} - {symbol}', red=True)
            else:
                self.update('How did we get here? Something went wrong.', red=True)
            return price_data

        price_data['date'] = [x.split(' ')[0] for x in price_data['date']]
        price_data.set_index('date', drop=True, inplace=True)

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

    def save_data(self, file_path):
        to_concat = [x for x in self.data + [self.prev_df] if not x.empty]
        if to_concat:
            pd.concat(to_concat, ignore_index=True).to_csv(file_path)

    def update(self, message, red=False):
        if red:
            message = f'\033[31m{message}\033[0m'

        with LOCK:
            print(f'({datetime.timedelta(seconds=int(round(time.time() - self.start)))}) {message}')


def main():
    collector = OptionsDataCollector(pd.read_csv(SYMBOLS_FILE_PATH))
    collector.run()


if __name__ == '__main__':
    main()
