from dataclasses import dataclass
from strategy import Strategy
from results import Results
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

@dataclass
class Backtester:
    
    data : pd.DataFrame
    weight_historic : np.ndarray[np.ndarray[float]] = None
    asset_values : np.ndarray[np.ndarray[float]] = None
    value_historic : np.ndarray[float] = None

    @property
    def backtest_length(self) -> int:
        return len(self.data)
    
    @property
    def nb_assets(self) -> int:
        return self.data.shape[1]
    
    def execute_backtest(self, initial_amount : float, strategy : Strategy) -> Results :
        
        '''Initialisation'''
        initial_weiths = np.full(self.nb_assets, 1/self.nb_assets)
        self.weight_historic = np.array([initial_weiths])
        self.asset_values = np.array([initial_weiths * initial_amount])
        self.value_historic = np.array(initial_amount)

        '''Itération pour chaque période du backtest'''
        for t in range(1, self.backtest_length + 1):
            
            previous_weights = self.weight_historic[t-1]
            daily_returns = np.array(self.data.iloc[t-1,:])
            
            new_asset_values = self.asset_values[t-1] * (1 + daily_returns)
            new_weights = strategy.compute_strat(previous_weights)

            self.asset_values = np.vstack([self.asset_values, new_asset_values])
            self.weight_historic = np.vstack([self.weight_historic, new_weights])
            self.value_historic = np.vstack([self.value_historic, np.sum(new_asset_values)])



def get_data(csv_path, start_date):

    df = pd.read_excel(csv_path)

    df_yf = pd.DataFrame()
    start_date = datetime.strptime(start_date, '%d/%m/%Y').strftime('%Y-%m-%d')

    for index, row in df.iterrows():
        ticker = row['Stock Ticker']
        stock_name = row['Stock Name']

        data = yf.download(ticker, start=start_date, progress=False)
        df_yf[stock_name] = data['Close']

    df_returns = df_yf.pct_change()
    df_returns = df_returns.drop(df_returns.index[0])

    return df_returns

df_returns = get_data('univers_actions.xlsx','01/01/2020')
df_returns = df_returns.iloc[200:300,0:5]

backtest = Backtester(df_returns)
backtest.execute_backtest(1000, Strategy())

perf_strat = list(backtest.value_historic)[1:]
dates = list(df_returns.index)
dates = [datetime.strftime(d, "%Y-%m-%d") for d in dates]
plt.plot(dates, perf_strat)
plt.show()