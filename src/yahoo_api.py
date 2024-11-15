from src.abstract_source import AbstractDataInput
from src.exeptions import FrequencyError
from src.tools import FrequencyType
from dataclasses import dataclass
import yfinance as yf
import pandas as pd

@dataclass
class YahooDataInput(AbstractDataInput):
    
    def _get_freq(self, frequency : str) -> str:
        """
        Map the frequency to YahooFinance accepted frequency str

        Args:
            frequency (str) : frequency selected by the user

        Returns:
            frequency (str) : Yahoo valid data frequency (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)
        """

        match frequency:
            case FrequencyType.MONTHLY:
                return '1mo'
            case FrequencyType.WEEKLY:
                return '1wk'
            case FrequencyType.DAILY:
                return '1d'
            case _:
                raise FrequencyError(f"Invalid frequency: {frequency}")

    def get_data(self, tickers : list[str], frequency : FrequencyType, start_date : str = '2023-10-01',  end_date : str = '2024-10-01') -> pd.DataFrame :
        """
        Retrieve the data related to the given tickers from Yahoo Finance API

        Args:
            tickers (list[str]) : YahooFinance format tickers
            start_date (str) : date of the first data
            end_date (str) : date of the last data
            frequency (str) : user input frequency

        Returns:
            df (pd.DataFrame) : price of every asset (ticker) at each date 
        """

        freq = self._get_freq(frequency)
        data : pd.DataFrame = yf.download(tickers, start=start_date, end=end_date, interval=freq, progress=False)['Adj Close']
        if isinstance(data, pd.DataFrame):
            data.reset_index(inplace=True)
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
            data["Date"] = data["Date"].apply(lambda x : x.strftime("%Y-%m-%d"))

        return data