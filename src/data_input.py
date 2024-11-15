from src.tools import InputType, FrequencyType, Index, Benchmark
from src.binance_api import BinanceDataInput
from src.custom_input import CustomDataInput
from src.df_input import DataFrameDataInput
from src.yahoo_api import YahooDataInput
from src.exeptions import InputTypeError
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import pandas as pd
    
@dataclass
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

    data_type : InputType 
    start_date : datetime = None
    end_date : datetime = None
    tickers : list[str] = None
    initial_weights : Optional[list[float]] = None
    frequency : FrequencyType = None
    index : Index = None
    file_path : str = None
    benchmark : Benchmark = None
    custom_df : pd.DataFrame = None

    @property
    def df_prices(self) -> pd.DataFrame:
        
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
            case _:
                raise InputTypeError("Unvalid asset price type selected")

        return data_requester.get_data(tickers=self.tickers,
                                        start_date=self.start_date,
                                        end_date=self.end_date,
                                        frequency=self.frequency)
    
    @property
    def df_benchmark(self) -> pd.Series:
        
        if self.benchmark is not None:
            ticker_bench = self.benchmark.value
            return YahooDataInput().get_data(tickers=ticker_bench,
                                            start_date=self.start_date,
                                            end_date=self.end_date,
                                            frequency=self.frequency)
        else :
            return None