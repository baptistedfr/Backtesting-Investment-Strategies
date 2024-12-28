from .abstract_source import AbstractDataInput
from ..exceptions import BadInput, InvalidFormat
from dataclasses import dataclass
import pandas as pd
import os.path

@dataclass
class DataFrameDataInput(AbstractDataInput):
    
    custom_df : pd.DataFrame

    def get_data(self, tickers, frequency, start_date : str = None,  end_date : str = None):
        """
        Run sanitary checks of the custom loaded dataframe

        Args:
            tickers (_type_): useless parameter, just here to keep the same structure as other AbtractSource Class children
            frequency (_type_): useless parameter, just here to keep the same structure as other AbtractSource Class children
            start_date (str, optional): strat date of the backtest
            end_date (str, optional): end date of the backtest
        """

        if "Date" not in self.custom_df:
            raise BadInput("'Date' column not in the selected dataframe")

        if len(self.custom_df.columns) < 2:
            raise BadInput("The input dataframe has to store at least one asset serie")
        
        if self.custom_df.isnull().values.any():
            raise BadInput("N/A found in selected dataframe")
        
        if start_date is not None and end_date is not None :
            self.custom_df['Date'] = pd.to_datetime(self.custom_df['Date'])
            self.custom_df = self.custom_df[(self.custom_df['Date'] >= start_date) & (self.custom_df['Date'] <= end_date)]

        return self.custom_df