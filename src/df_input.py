from src.abstract_source import AbstractDataInput
from src.exeptions import BadInput, InvalidFormat
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
        df = self.custom_df.copy()

        if "Date" not in df.columns:
            raise BadInput("'Date' column not in the selected dataframe")

        if len(df.columns) < 2:
            raise BadInput("The input dataframe has to store at least one asset serie")
        
        if df.isnull().values.any():
            raise BadInput("N/A found in selected dataframe")
        
        if start_date is not None and end_date is not None :
            if start_date not in df["Date"] or end_date not in df["Date"]:
                raise BadInput("Selected start or end date not present in dataframe")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        return df