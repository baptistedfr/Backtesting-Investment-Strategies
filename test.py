# from apis.yahoo_api import YahooFinanceApi
# import pandas as pd
# from datetime import datetime

# df = pd.read_excel("data/cac40.xlsx")
# tickers_list = list(set(df['Ticker']))

# yf = YahooFinanceApi()
# data = yf.get_data(tickers=tickers_list)
# print(data.head())

# from apis.binance_api import BinanceApi
# import pandas as pd

# df = pd.read_excel("data/crypto_data.xlsx")
# tickers_list  = list(df['Ticker'].values)[:3]

# binance_reader = BinanceApi()
# binance_data = binance_reader.get_data(ticker_list= tickers_list, frequency = "D", colums_select = ['Close time','Close'])
# print(binance_data)

from src.data_input import DataInput, InputType
import numpy as np
import pandas as pd

# df_prices = get_data('univers_actions.xlsx','01/01/2020')
# print(df_prices.head())
# df_prices = df_prices.iloc[200:800,0:5]
# df_benchmark = get_benchmark('01/01/2020')
# df_benchmark = df_benchmark[200:800]

df = pd.read_excel("data/cac40.xlsx")
tickers_list = list(set(df['Ticker']))

data = DataInput(asset_type=InputType.EQUITY,
                 tickers=tickers_list,
                 start_date='2023-10-01',
                 end_date='2024-10-01',
                 frequency="D")
print("end")