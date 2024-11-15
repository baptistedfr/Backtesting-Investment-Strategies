from src.abstract_source import AbstractDataInput
from src.exeptions import FrequencyError
from src.tools import FrequencyType
from binance.client import Client
from dataclasses import dataclass
import pandas as pd

@dataclass
class BinanceDataInput(AbstractDataInput):

    def _get_freq(self, frequency : str):
        """
        Map the input frequency to YahooFinance accepted frequency str

        Args:
            frequency (str) : frequency selected by the user

        Returns:
            frequency : Binance valid data frequency
        """
        match frequency:
            case FrequencyType.MONTHLY:
                return Client.KLINE_INTERVAL_1MONTH
            case FrequencyType.WEEKLY:
                return Client.KLINE_INTERVAL_1WEEK
            case FrequencyType.DAILY:
                return Client.KLINE_INTERVAL_1DAY
            case _:
                raise FrequencyError(f"Invalid frequency: {frequency}")

    def _retreat_results(self,  df_binance : pd.DataFrame, ticker : str,
                         columns_selected : list[str] = ['Close time','Close']) -> pd.DataFrame :
        '''
        Retreat Binance informations to set the right format : numeric columns, date format and columns renamed

        Args:
            df_binance (pd.DataFrame) : dataframe retrieved from Binance
            ticker (str) : ticker of the cryptocurrency
            columns_selected (list[str]) : columns to keep in the Binance dataframe

        Returns:
            df (pd.DataFrame) : dataframe retreaded
        '''

        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                   'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                   'Taker buy quote asset volume', 'Ignore']

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                           'Number of trades', 'Taker buy base asset volume',
                           'Taker buy quote asset volume']

        df = pd.DataFrame(df_binance, columns=columns)
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        df['Close time'] = df["Close time"].apply(lambda x : x.strftime("%Y-%m-%d"))
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        df = df[columns_selected]
        ticker_cleaned = ticker.replace("USDT", "")
        df.rename(columns={"Close":ticker_cleaned, "Close Time":"Date"}, inplace = True)
        
        return df

    def get_data(self, tickers : list[str], start_date : str = '2023-10-01',  end_date : str = '2024-10-01', 
                 frequency : str = "D", colums_select : list =['Close time','Close']) -> pd.DataFrame :
        """
        Request Binance API to retrieve prices on selected tickers over the selected period

        Args:
            ticker_list (list[str]): tickers to request Binance
            start_date_str (str, optional): Defaults to "1 Jan, 2019".
            end_date_str (str, optional): Defaults to "1 Jan, 2024".
            frequency (str, optional): Defaults to "D".
            colums_select (list, optional): Defaults to ['Close time','Close'].

        Returns:
           df (pd.DataFrame) : Retreated prices for selected tickers
        """

        client = Client()

        freq = self._get_freq(frequency)

        data_final = pd.DataFrame()
        for ticker in tickers:

            result_binance = []
            for k_line in client.get_historical_klines_generator(symbol=ticker,
                                                                interval=freq,
                                                                start_str=start_date, 
                                                                end_str=end_date):
                result_binance.append(k_line)

            results_retreated = self._retreat_results(result_binance, ticker, colums_select)

            if data_final.empty:
                data_final = results_retreated
            else:
                data_final = data_final.merge(results_retreated, on="Close time", how="left")

        return data_final.rename(columns={"Close time":"Date"})