from src.abstract_source import AbstractDataInput
from src.exeptions import BadInput, InvalidFormat
from dataclasses import dataclass
import pandas as pd
import os.path

@dataclass
class CustomDataInput(AbstractDataInput):
    
    file_path : str

    def get_data(self, tickers, frequency, start_date : str = None,  end_date : str = None):
        """
        Load the selected file in a dataframe and run sanitary checks

        Args:
            file_path (str): path of the file to open
            tickers (_type_): useless parameter, just here to keep the same structure as other AbtractSource Class children
            frequency (_type_): useless parameter, just here to keep the same structure as other AbtractSource Class children
            start_date (str, optional): strat date of the backtest
            end_date (str, optional): end date of the backtest

        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("File doesn't exist in the current folder")
        
        file_sufix = self.file_path.split('.')[-1]
        match file_sufix:
            case 'csv':
                df = pd.read_csv(self.file_path, sep=';')
            case 'xlsx':
                df = pd.read_excel(self.file_path)
            case 'parquet':
                df = pd.read_parquet(self.file_path)
            case _:
                raise InvalidFormat('File format not in : [csv, xlsx, parquet]')

        if "Date" not in df.columns:
            raise BadInput("'Date' column not in the selected file")
        
        if len(df.columns) < 2:
            raise BadInput("The input file has to store at least one asset serie")
        
        if df.isnull().values.any():
            raise BadInput("N/A found in selected file")
        
        if start_date is not None and end_date is not None :
            if start_date not in df["Date"] or end_date not in df["Date"]:
                raise BadInput("Selected start or end date not present in file")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        return df