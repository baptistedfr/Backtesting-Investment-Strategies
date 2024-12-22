from dataclasses import dataclass
from typing import *
from datetime import datetime
import pandas as pd
from functools import cached_property
from .tools import InputType, FrequencyType, Index, Benchmark
#from .data_inputs import BinanceDataInput, CustomDataInput, DataFrameDataInput, YahooDataInput, BLPApi
from .data_inputs import BinanceDataInput, CustomDataInput, DataFrameDataInput, YahooDataInput
from .exceptions import InputTypeError, BadInput

class DataInput:
    """
    Generic interface between the APIs & the backtester, stores the prices dataframe as an attribute

    Args:
        data_type (Enum InputType) : type of input (APIs or custom files)
        start_date (optional datetime) : start of the backtest period
        end_date (optional datetime) : start of the backtest period
        tickers (optional list[str]) : tickers to retreave from the API service
        frequency (optional Enum FrequencyType) : frequency of the Data
        index (optional Enum Index) : asset to extract the composition from
        file_path (optional str) : path of the custom file
        benchmark (optional Index) : benchmark to compare the performance of the strategy
        custom_df (optional pd.DataFrame) : custom dataframe as a data input

        df_prices (pd.DataFrame) : asset prices
    """
    def __init__(self, data_type : InputType, 
                 frequency : FrequencyType,
                 start_date : datetime = None, 
                 end_date : datetime = None, 
                 tickers : list[str] = None, 
                 initial_weights : Optional[list[float]] = None, index : Index = None,
                 file_path : str = None, 
                 custom_df : pd.DataFrame = None, 
                 benchmark : Benchmark = None):
        
        self.data_type : InputType = data_type
        self.start_date : datetime = start_date
        self.end_date : datetime = end_date
        self.tickers : list[str] = tickers
        self.initial_weights : Optional[list[float]] = initial_weights 
        self.frequency : FrequencyType = frequency
        self.index : Index = index
        self.file_path : str = file_path
        self.custom_df : pd.DataFrame = custom_df
        self.benchmark : Benchmark = benchmark
        
        self.df_prices : pd.DataFrame = self.get_prices()
        if (benchmark is not None and (start_date is None or end_date is None)):
            try:
                self.start_date= self.df_prices.at[0,'Date'].strftime("%Y-%m-%d")
                self.end_date= self.df_prices['Date'].iloc[-1].strftime("%Y-%m-%d")
            except:
                raise BadInput("La date de début et de fin n'est pas reconnaissable")
        
        self.df_benchmark : pd.DataFrame = self.get_benchmark()
        
        if (self.df_benchmark is not None) and (len(self.df_prices) != len(self.df_benchmark)):
            print("Attention, le benchmark et les datas n'ont pas le meme nombre de lignes, nous allons conserver les lignes en commun!\n")
            print('Taille des datas:',len(self.df_prices))
            print('Taille du benchmark:',len(self.df_benchmark))
            self.df_prices = self.df_prices[self.df_prices['Date'].isin(self.df_benchmark['Date'].unique())]
            self.df_benchmark = self.df_benchmark[self.df_benchmark['Date'].isin(self.df_prices['Date'].unique())]
            self.df_prices = self.df_prices.sort_values(by='Date').reset_index(drop=True)
            self.df_benchmark = self.df_benchmark.sort_values(by='Date').reset_index(drop=True)

    def get_prices(self) -> pd.DataFrame:
        match self.data_type:
            case InputType.FROM_FILE:
                data_requester = CustomDataInput(self.file_path)
            case InputType.FROM_DATAFRAME:
                data_requester = DataFrameDataInput(self.custom_df)
            case InputType.EQUITY:
                data_requester = YahooDataInput()
            case InputType.CRYPTO:
                data_requester = BinanceDataInput()
            case InputType.FROM_INDEX_COMPOSITION:
                data_requester = YahooDataInput()
                path_index = "data/" + self.index.value + ".xlsx"
                index_composition = pd.read_excel(path_index)
                self.tickers = list(set(index_composition['Ticker']))
                if ("Poids" in index_composition.columns):
                    self.initial_weights = list(index_composition['Poids'])
            case InputType.FROM_BLOOMBERG:
                data_requester = BLPApi()
            case _:
                raise InputTypeError("Unvalid asset price type selected")

        return data_requester.get_data(tickers=self.tickers,
                                        start_date=self.start_date,
                                        end_date=self.end_date,
                                        frequency=self.frequency)
    
    '''def get_benchmark(self) -> pd.Series:
        
        if self.benchmark is not None:
            category_bench = self.benchmark.category
            ticker_bench = [self.benchmark.symbol]
            if category_bench == "Equity":
                if self.data_type == InputType.FROM_BLOOMBERG:
                    return BLPApi().get_data(tickers=ticker_bench,
                                            start_date=self.start_date,
                                            end_date=self.end_date,
                                            frequency=self.frequency)
                else:
                    return YahooDataInput().get_data(tickers=ticker_bench,
                                            start_date=self.start_date,
                                            end_date=self.end_date,
                                            frequency=self.frequency)
            else:
                return BinanceDataInput().get_data(tickers=ticker_bench,
                                            start_date=self.start_date,
                                            end_date=self.end_date,
                                            frequency=self.frequency)
        else :
            return None'''

    def get_benchmark(self) -> pd.Series:

        if self.benchmark is not None:
            category_bench = self.benchmark.category
            ticker_bench = [self.benchmark.symbol]
            if category_bench == "Equity":
                return YahooDataInput().get_data(tickers=ticker_bench,
                                                 start_date=self.start_date,
                                                 end_date=self.end_date,
                                                 frequency=self.frequency)
            else:
                return BinanceDataInput().get_data(tickers=ticker_bench,
                                                   start_date=self.start_date,
                                                   end_date=self.end_date,
                                                   frequency=self.frequency)
        else:
            return None