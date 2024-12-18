from enum import Enum

class InputType(Enum):
    EQUITY = "Equity"
    CRYPTO = "Crypto"
    FROM_INDEX_COMPOSITION = "Index"
    FROM_FILE = "File"
    FROM_DATAFRAME = "DataFrame"
    FROM_BLOOMBERG = "Bloomberg"

class FrequencyType(Enum):
    DAILY = 252 # 252 jours de trading dans une année
    WEEKLY = 52 # 52 semaines dans une année
    MONTHLY = 12 # 12 mois dans une année

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
    BTC = ("Crypto", "BTCUSDT")
    ETH = ("Crypto", "ETHUSDT")
    SOL = ("Crypto","SOLUSDT")
    
    def __init__(self, category, symbol):
        self.category = category
        self.symbol = symbol