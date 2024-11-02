import pandas as pd
from datetime import datetime
from backtester import Backtester
from strategy import RandomFluctuationStrategy, FocusedStrategy
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

def get_data(csv_path, start_date):

    df = pd.read_excel(csv_path)

    df_yf = pd.DataFrame()
    start_date = datetime.strptime(start_date, '%d/%m/%Y').strftime('%Y-%m-%d')

    for index, row in df.iterrows():
        ticker = row['Stock Ticker']
        stock_name = row['Stock Name']

        data = yf.download(ticker, start=start_date, progress=False)
        df_yf[stock_name] = data['Close']

    return df_yf

df_prices = get_data('univers_actions.xlsx','01/01/2020')
df_prices = df_prices.iloc[200:,0:5]

strategy = FocusedStrategy(0)
initial_weights = np.full(5, 0)
print(initial_weights)
initial_weights[0] = 1.0
backtest = Backtester(df_prices=df_prices, initial_amount=1000.0, strategy=strategy, initial_weights=initial_weights)
backtest.run()

print(backtest.ptf_values.head())
print(backtest.ptf_weights.head())
print("End")

perf_strat = list(backtest.ptf_values)
dates = list(df_prices.index)
dates = [datetime.strftime(d, "%Y-%m-%d") for d in dates]
plt.plot(dates, perf_strat)
plt.show()