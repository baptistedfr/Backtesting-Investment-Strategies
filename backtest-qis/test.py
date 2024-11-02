from strategy import RandomFluctuationStrategy, FocusedStrategy
from backtester import Backtester
from tools import get_data
import numpy as np

df_prices = get_data('univers_actions.xlsx','01/01/2020')
df_prices = df_prices.iloc[:,0:5]

# strategy = FocusedStrategy(0)
# initial_weights = np.full(5, 0)
# initial_weights[0] = 1.0

strategy = RandomFluctuationStrategy()
backtest = Backtester(df_prices=df_prices, initial_amount=1000.0, strategy=strategy)
results = backtest.run()

print(results.statistics.head(10))
plot = results.strat_plot
plot.show()