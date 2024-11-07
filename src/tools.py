import time
import pandas as pd
from datetime import datetime
import yfinance as yf
from enum import Enum

def timer(func):
    """Decorator used to compute the execution time of a method"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__qualname__} took {elapsed_time:.4f} sec to execute")
        return result
    return wrapper

def get_benchmark(start_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    data = yf.download("^FCHI", start=start_date, progress=False)
    series_cac40 = pd.Series(data['Close'], name='CAC40')
    return series_cac40

class InputType(Enum):
    CUSTOM = "Custom"
    EQUITY = "Equity"
    CRYPTO = "Crypto"
    FROM_INDEX_COMPOSITION = "Index"

class FrequencyType(Enum):
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"

class Index(Enum):
    CAC40 = "cac40"
    STX50 = "eurostoxx50"
    NIKKEI = "nikkei"
    SP500 = "sp500"

class Benchmark(Enum):
    CAC40 = "^FCHI"
    DAX = "^GDAXI"
    FTSE100 = "^FTSE"
    SP500 = "^GSPC"
    NASDAQ = "^IXIC"
    NIKKEI225 = "^N225"
    HANGSENG = "^HSI"