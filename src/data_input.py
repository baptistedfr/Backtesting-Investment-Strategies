from src.binance_api import BinanceApi
from src.yahoo_api import YahooFinanceApi
from src.exeptions import InputError
from dataclasses import dataclass
from src.tools import InputType, FrequencyType
from datetime import datetime
from enum import Enum
import pandas as pd
    
@dataclass
class DataInput:

    asset_type : InputType 
    tickers : list[str]
    start_date : datetime
    end_date : datetime
    frequency : FrequencyType

    @property
    def import_data(self) -> pd.DataFrame:
        
        match self.asset_type:
            case InputType.CUSTOM:
                data_requester = ...
            case InputType.EQUITY:
                data_requester = YahooFinanceApi()
            case InputType.CRYPTO:
                data_requester = BinanceApi()
            case _:
                raise InputError("Unvalid asset price type selected")

        return data_requester.get_data(tickers=self.tickers,
                                        start_date=self.start_date,
                                        end_date=self.end_date,
                                        frequency=self.frequency)