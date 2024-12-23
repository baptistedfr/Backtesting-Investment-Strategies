from .abstract_source import AbstractDataInput
from ..exceptions import FrequencyError, DataError
from ..tools import FrequencyType
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

    def get_data(self, tickers : list[str], frequency : FrequencyType, start_date : str = '2023-10-01',  end_date : str = '2024-10-01') -> pd.DataFrame:
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
        data : pd.DataFrame = yf.download(tickers, start=start_date, end=end_date, interval=freq, progress=False)['Close']
        if isinstance(data, pd.DataFrame):
            data.reset_index(inplace=True)
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
            data["Date"] = data["Date"].apply(lambda x : x.strftime("%Y-%m-%d"))
            data['Date']=pd.to_datetime(data['Date'])
        for ticker in tickers:
            if pd.isnull(data.loc[0, ticker]):
                print(f"The ticker {ticker} does not have a value in {start_date}")
        return data

    def get_PER(self, tickers: list[str], start_date: str = '2023-10-01', end_date: str = '2024-10-01', frequency: FrequencyType = FrequencyType.DAILY) -> pd.DataFrame:
        """
        Retrieve the trailing PE for each ticker from Yahoo Finance API

        Args:
            tickers (list[str]) : YahooFinance format tickers
            start_date (str) : start date of the data
            end_date (str) : end date of the data
            frequency (str) : user input frequency (optional)

        Returns:
            pd.DataFrame : DataFrame with tickers as columns and their trailing PE as values
        """
        #freq = self._get_freq(frequency)

        # Retrieve trailing PE values for each ticker
        per_dict = {}

        data = self.get_data(tickers, frequency, start_date, end_date)

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            trailing_pe = info.get('trailingPE', None)  # Trailing PE

            # For each ticker, repeat the trailing PE value across all dates in the range
            per_dict[ticker] = [trailing_pe] * len(data)  # Repeating the same value across dates

            # Convert the dictionary to a DataFrame
        per_df = pd.DataFrame(per_dict, index=data.index)

        '''# Apply rolling window if there are any missing values (if necessary)
        if per_df.isnull().values.any():
            per_df = per_df.rolling(window=rolling_window, min_periods=1).mean()'''

        return per_df