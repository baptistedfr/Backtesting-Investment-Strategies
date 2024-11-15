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

class InputType(Enum):
    EQUITY = "Equity"
    CRYPTO = "Crypto"
    FROM_INDEX_COMPOSITION = "Index"
    FROM_FILE = "File"
    FROM_DATAFRAME = "DataFrame"

class FrequencyType(Enum):
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12

class Index(Enum):
    CAC40 = "cac40"
    STX50 = "eurostoxx50"
    NIKKEI = "nikkei"
    SP500 = "sp500"

class Benchmark(Enum):
    # Equity benchmarks
    CAC40 = ("Equity", "^FCHI")
    DAX = ("Equity", "^GDAXI")
    FTSE100 = ("Equity", "^FTSE")
    SP500 = ("Equity", "^GSPC")
    NASDAQ = ("Equity", "^IXIC")
    NIKKEI225 = ("Equity", "^N225")
    HANGSENG = ("Equity", "^HSI")
    
    # Crypto benchmarks
    BTC = ("Crypto", ["BTCUSDT"])
    ETH = ("Crypto", ["ETHUSDT"])

    def __init__(self, category, symbol):
        self.category = category
        self.symbol = symbol