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
SYMBOLS_FILE_PATH = 'venv/tickers/s&p-500.csv'
START_DATE = datetime.datetime.strptime('2001-11-01', DATE_FORMAT)
END_DATE = datetime.datetime.today()
OPTION_TYPE = 'C'
SKIP_DAYS = 14
MAX_THREADS = 10

lock = threading.Lock()


def process_find_options(find_options, symbol, date, option_symbols, option_symbols_to_process, finished_symbols):
    with lock:
        if symbol in finished_symbols and datetime.datetime.strptime(date, DATE_FORMAT) < datetime.datetime.strptime(
                finished_symbols[symbol], DATE_FORMAT):
            return

    results = find_options(symbol=symbol, date=date, callPut=OPTION_TYPE)

    if results.empty and symbol:
        if symbol not in finished_symbols:
            with lock:
                print('Finished symbol:', symbol, date)
                finished_symbols[symbol] = date
        return

    for symbol, exp_date in zip(results['optionSymbol'], results['expirationDate']):
        if symbol not in option_symbols:
            with lock:
                option_symbols.add(symbol)
                option_symbols_to_process[symbol] = exp_date


def process_get_options_data(get_options_data, option_symbol, exp_date_str, options_data, price_data):
    exp_date = datetime.datetime.strptime(exp_date_str, DATE_FORMAT)
    if exp_date > END_DATE:
        return

    for days in [28, 14, 7]:
        from_date_str = (exp_date - datetime.timedelta(days=days)).strftime(DATE_FORMAT)
        results = get_options_data(symbol=option_symbol, from_=from_date_str, to=exp_date_str)
        if 'error' in results.columns:
            results = pd.DataFrame()
        if not results.empty:
            break

    if not results.empty:
        new_data = []
        for symbol, date in results[['symbol', 'expiration']].values:
            if symbol is None or date is None:
                new_data.append([None, None, None])
                continue

            attempts = 0
            while attempts < 5:
                try:
                    new_data.append([price_data['open'][date], price_data['close'][date], attempts])
                except KeyError:
                    date = (datetime.datetime.strptime(date, DATE_FORMAT) - datetime.timedelta(days=1)).strftime(
                        DATE_FORMAT)
                    attempts += 1
                else:
                    break
            else:
                new_data.append([None, None, None])

        results[['open_expiry_price', 'close_expiry_price', 'days_behind']] = new_data

    with lock:
        options_data[option_symbol] = results


def main():
    start = time.time()
    print(f'({datetime.timedelta(seconds=round(time.time() - start))}) Starting to collect options data...')

    ivol.set_login_params(username='BCInvest', password='Investpass123!')
    find_options = ivol.set_method('/equities/eod/option-series-on-date')
    get_options_data = ivol.set_method('/equities/eod/single-stock-option-raw-iv')
    get_price_data = ivol.set_method('/equities/eod/stock-prices')

    nyse_dates = xcal.get_calendar(
        'NASDAQ', start=START_DATE.strftime(DATE_FORMAT), end=END_DATE.strftime(DATE_FORMAT)).schedule['open']
    nasdaq_dates = xcal.get_calendar(
        'NASDAQ', start=START_DATE.strftime(DATE_FORMAT), end=END_DATE.strftime(DATE_FORMAT)).schedule['open']
    dates = list(sorted(set(list(nyse_dates) + list(nasdaq_dates))))

    symbols = pd.read_csv(SYMBOLS_FILE_PATH)
    option_symbols = set()
    finished_symbols = {}
    options_data = {}
    futures = {}

    prev_df = pd.DataFrame()  # pd.read_parquet('options-data/100-1700017473.parquet')
    symbols_done = set(prev_df['symbol']) if prev_df is not None and not prev_df.empty else set()
    symbols_to_do = [symbol for symbol in symbols['Symbol'] if symbol not in symbols_done]

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for i, symbol in enumerate(symbols_to_do, 1):
            with lock:
                option_symbols_to_process = {}
                print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) '
                      f'({i:,} / {len(symbols):,}) {symbol}')

            last_date = None
            for j, date in enumerate(reversed(dates), 1):
                if j % 10_000 == 0:
                    with lock:
                        print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) '
                              f'({j:,} / {len(dates):,}) dates processed')

                if (last_date is not None) and (last_date - date) <= datetime.timedelta(days=SKIP_DAYS):
                    continue
                last_date = date

                date_str = date.strftime('%Y-%m-%d')
                future = executor.submit(process_find_options, find_options, symbol, date_str, option_symbols,
                                         option_symbols_to_process, finished_symbols)
                futures[future] = (symbol, date_str)

                if len(futures) == MAX_THREADS * 10:
                    for future in as_completed(futures):
                        futures.pop(future)
                        if future.exception():
                            print(future.exception())
                        if len(futures) == MAX_THREADS * 9:
                            break

            with lock:
                print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) '
                      f'({j:,} / {len(dates):,}) dates processed')
                print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) Gathering price data..')

            price_data = get_price_data(symbol=symbol, from_=START_DATE.strftime(DATE_FORMAT),
                                        to=END_DATE.strftime(DATE_FORMAT))
            price_data['date'] = [x.split(' ')[0] for x in price_data['date']]
            price_data.set_index('date', drop=True, inplace=True)

            with lock:
                print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) Gathering options data..')

            for j, (option_symbol, exp_date) in enumerate(option_symbols_to_process.copy().items(), 1):
                if j % 1_000 == 0:
                    with lock:
                        print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) '
                              f'({j:,} / {len(option_symbols_to_process):,}) option symbols processed')

                future = executor.submit(process_get_options_data, get_options_data, option_symbol, exp_date, options_data,
                                         price_data)
                futures[future] = option_symbol

                if len(futures) == MAX_THREADS * 10:
                    for future in as_completed(futures):
                        futures.pop(future)
                        if future.exception():
                            print(future.exception())
                        if len(futures) == MAX_THREADS * 9:
                            break

            size = sys.getsizeof(options_data) / 1024 ** 2
            print(f'({datetime.timedelta(seconds=int(round(time.time() - start)))}) Saving {size:,.2f}MB of data...')
            file_path = f'venv/options-data/{len(symbols)}-{int(start)}.parquet'
            with lock:
                to_concat = []
                if prev_df is not None and not prev_df.empty:
                    to_concat.append(prev_df)
                to_concat.extend([x for x in options_data.values()])
                df = pd.concat(to_concat, ignore_index=True)
            df.to_parquet(file_path, engine='fastparquet')


if __name__ == '__main__':
    main()
