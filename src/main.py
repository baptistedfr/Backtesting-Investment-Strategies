from strategy import RandomFluctuationStrategy, FocusedStrategy
from src.data_input import DataInput, InputType
from tools import get_data, get_benchmark
from backtester import Backtester
import numpy as np
import pandas as pd

# df_prices = get_data('univers_actions.xlsx','01/01/2020')
# print(df_prices.head())
# df_prices = df_prices.iloc[200:800,0:5]
# df_benchmark = get_benchmark('01/01/2020')
# df_benchmark = df_benchmark[200:800]

df = pd.read_excel("data/cac40.xlsx")
tickers_list = list(set(df['Ticker']))

data = DataInput(asset_type=InputType.EQUITY,
                 tickers=tickers_list,
                 start_date='2023-10-01',
                 end_date='2024-10-01',
                 frequency="D")

print("End")
# strategy = FocusedStrategy(0)
# initial_weights = np.full(5, 0)
# initial_weights[0] = 1.0
# backtest = Backtester(df_prices=df_prices, initial_amount=1000.0, strategy=strategy)
# results_focused = backtest.run()

# strategy = RandomFluctuationStrategy()
# backtest = Backtester(df_prices=df_prices, initial_amount=1000.0, strategy=strategy)
# results_random = backtest.run()

# results_compared = results_focused.compare_with(other=results_random, name_self="Focused", name_other="Random")

# print(results_compared.df_statistics.head(10))
# results_compared.ptf_value_plot.show()
# results_compared.ptf_weights_plot.show()
# results_compared.other_weights_plot.show()