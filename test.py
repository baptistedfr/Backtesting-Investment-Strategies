from apis.yahoo_api import YahooFinanceApi
import pandas as pd
from datetime import datetime

df = pd.read_excel("data/cac40.xlsx")
tickers_list = list(set(df['Ticker']))

yf = YahooFinanceApi()
data = yf.get_data(tickers=tickers_list)
print(data.head())

from apis.binance_api import BinanceApi
import pandas as pd

df = pd.read_excel("data/crypto_data.xlsx")
tickers_list  = list(df['Ticker'].values)[:3]

binance_reader = BinanceApi()
binance_data = binance_reader.get_data(ticker_list= tickers_list, frequency = "D", colums_select = ['Close time','Close'])
print(binance_data)