import time
import pandas as pd
from datetime import datetime
import yfinance as yf

def timer(func):
    """Decorator used to compute the execution time of a method"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__qualname__} took {elapsed_time:.4f} sec to execute")
        return result
    return wrapper

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

def get_benchmark(start_date):
    start_date = datetime.strptime(start_date, '%d/%m/%Y').strftime('%Y-%m-%d')
    data = yf.download("^FCHI", start=start_date, progress=False)
    series_cac40 = pd.Series(data['Close'], name='CAC40')
    return series_cac40