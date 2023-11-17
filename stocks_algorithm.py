import os
import pandas as pd
import numpy as np
import yfinance as yf

DATA_FOLDER = 'stock-data'


def generate_file_path(ticker, start_date, end_date):
    return f'{DATA_FOLDER}/{ticker}_{start_date}_{end_date}.csv'


class StockData:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.path = generate_file_path(ticker, start_date, end_date)
        self.df = None
        self.get_data()

    def __getattr__(self, attr):
        return getattr(self.df, attr)

    def get_data(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path, index_col=0)
            self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
        else:
            self.df = yf.download(self.ticker, self.start_date, self.end_date)
            self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
            self.df.to_csv(self.path)



def main():
    ticker = 'SPY'
    start_date = '1993-01-29'
    end_date = '2023-11-10'
    data = StockData(ticker, start_date, end_date)
    # simulation = Simulation(data)
    # simulation.run()
    # simulation.update()

    quotes = data.df['Adj Close']
    trades = []
    total_roi = 1

    for i, quote in enumerate(quotes):
        if i > 3 and (quotes.iloc[i - 1] - quotes.iloc[i - 2]) / quotes.iloc[i - 2] < 0.013 and (
                quotes.iloc[i - 2] - quotes.iloc[i - 3]) / quotes.iloc[i - 3] < 0.013:
            trades.append(quote / quotes.iloc[i - 1] - 1)
            total_roi *= quote / quotes.iloc[i - 1]

    average = sum(trades) / len(trades)
    std = np.std(trades)
    ratio = len(trades) / len(quotes)
    roi = total_roi ** (252 / len(quotes)) - 1
    baseline_roi = (quotes.iloc[-1] / quotes.iloc[0]) ** (252 / len(quotes)) - 1

    print(
        f'Average Trade ROI: {average * 100:,.4f}% | Trade ROI STD: {std * 100:,.4f}% | Trade Ratio: {ratio * 100:.2f}% | Annualized ROI: {roi * 100:,.2f}% | Baseline Annualized ROI: {baseline_roi * 100:,.2f}%')


if __name__ == '__main__':
    main()
