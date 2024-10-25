import pandas as pd
from dataclasses import dataclass
import yfinance as yf

@dataclass
class YahooFinanceApi:
    
    def _get_freq(self, frequence):
        '''
        interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        '''
        match frequence:
            case "3M":
                return '3mo'
            case "M":
                return '1mo'
            case "W":
                return '1wk'
            case "D":
                return '1d'
            case "H":
                return '1h'
            case _:
                raise ValueError(f"Invalid frequency: {frequence}")

    def get_data(self, ticker_list : list[str], start_date : str = '2023-10-01',  end_date : str = '2024-10-01', 
                 frequence : str = "M") -> pd.DataFrame :
        '''
        Retrieve the data related to the given tickers from Yahoo Finance API
        '''
        freq = self._get_freq(frequence)
        data = yf.download(ticker_list, start=start_date, end=end_date, interval=freq)['Adj Close']
        data.reset_index(inplace=True)
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
        return data