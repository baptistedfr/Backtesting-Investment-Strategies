from dataclasses import dataclass
from binance.client import Client
import pandas as pd

@dataclass
class BinanceApi:
    '''
    Class used to retrieve data from Binance API with a given set of parameters
    '''
    
    binance_client : Client = Client()

    def _get_freq(self, frequence):
        '''
        Get the Binance frequence corresponding to the user input 
        '''
        match frequence:
            case "M":
                return Client.KLINE_INTERVAL_1MONTH
            case "W":
                return Client.KLINE_INTERVAL_1WEEK
            case "D":
                return Client.KLINE_INTERVAL_1DAY
            case "H":
                return Client.KLINE_INTERVAL_1HOUR
            
            case _:
                raise ValueError(f"Invalid frequency: {frequence}")

    def _retreat_results(self,  result_binance : pd.DataFrame, columns_select : list) -> pd.DataFrame :
        '''
        Retreat the info get from Binance in the correct format
        '''
        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
            'Taker buy quote asset volume', 'Ignore'
        ]

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                            'Number of trades', 'Taker buy base asset volume',
                            'Taker buy quote asset volume']

        df = pd.DataFrame(result_binance, columns=columns)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        return df[columns_select]

    def get_data(self, ticker_list : list[str], start_date_str : str = "1 Jan, 2019",  end_date_str : str = "1 Jan, 2024", 
                 frequence : str = "M", colums_select : list =['Close time','Close']) -> pd.DataFrame :
        '''
        Retrieve the data related to the given tickers from Binance API
        '''
        freq = self._get_freq(frequence)
        data_final = pd.DataFrame()
        for ticker in ticker_list:

            result_binance = []
            for k_line in self.binance_client.get_historical_klines_generator(symbol=ticker,
                                                                        interval=freq,
                                                                        start_str=start_date_str, end_str=end_date_str):
                result_binance.append(k_line)
            results_retreated = self._retreat_results(result_binance, colums_select)
            ticker_cleaned = ticker.replace("USDT", "")
            results_retreated.rename(columns={"Close":ticker_cleaned}, inplace = True)
            if data_final.empty:
                data_final = results_retreated
            else:
                data_final = data_final.merge(results_retreated, on="Close time", how="left")

        return data_final