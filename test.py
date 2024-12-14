from src.tools import InputType, FrequencyType, Index, Benchmark
from src.strategy import (RandomFluctuationStrategy, FocusedStrategy, TrendFollowingStrategy, MomentumStrategy,
                          LowVolatilityStrategy)
from src.backtester import Backtester
from src.data_input import DataInput
from src.results import Results
import pandas as pd

# data = DataInput(data_type=InputType.EQUITY,
#                         tickers=['GLE.PA', 'OR.PA','MC.PA'],
#                         start_date='2023-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.DAILY,
#                         benchmark=Benchmark.CAC40)

data = DataInput(data_type=InputType.CRYPTO,
                        tickers=['BTCUSDT','ETHUSDT','PEPEUSDT','DOGEUSDT','SOLUSDT'],
                        start_date='2018-10-01',
                        end_date='2024-11-15',
                        frequency=FrequencyType.WEEKLY,
                        benchmark = Benchmark.BTC)


# data = DataInput(data_type=InputType.FROM_FILE,
#                         file_path='data/custom.xlsx')

# print(data.df_prices)
# df = pd.read_excel("data/custom.xlsx")
# data = DataInput(data_type=InputType.FROM_DATAFRAME,
#                 custom_df=df,
#                 benchmark=Benchmark.CAC40,
#                 frequency=FrequencyType.DAILY)



# data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
#                 index=Index.CAC40,
#                 start_date='2010-10-01',
#                 end_date='2024-10-01',
#                 frequency=FrequencyType.WEEKLY,
#                 benchmark=Benchmark.CAC40)



# # prices = data.df_prices
strategy = RandomFluctuationStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=0)
backtest = Backtester(data_input=data, custom_name="Fees=0.1%")

# # # # weight = backtest.initial_weights_value
# # # # print(weight)
results_random = backtest.run(strategy=strategy, initial_amount=1000.0, fees=0.001)

strategy2 = RandomFluctuationStrategy(rebalance_frequency = FrequencyType.WEEKLY, lookback_period=1)
backtest2 = Backtester(data_input=data, custom_name="No Fees")
results_random2 = backtest2.run(strategy=strategy2, initial_amount=1000.0, fees=0.0)

combined_results = Results.compare_results([results_random, results_random2])
print(combined_results.df_statistics.head(10))
combined_results.ptf_value_plot.show()
combined_results.ptf_drawdown_plot.show()
for plot in combined_results.ptf_weights_plot:
    plot.show()

# data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
#                 index=Index.CAC40,
#                 start_date='2023-05-01',
#                 end_date='2024-11-01',
#                 frequency=FrequencyType.WEEKLY,
#                 benchmark=Benchmark.CAC40)

# strategy = MeanRevertingStrategy()
# backtest = Backtester(data_input=data, custom_name="Mean Reverting - No Fees")
# results_random = backtest.run(strategy=strategy, initial_amount=1000.0, fees=0.0, delayed_start="2023-05-29")

# strategy2 = LowVolatilityStrategy()
# backtest2 = Backtester(data_input=data, custom_name="Low Vol - No Fees")
# results_random2 = backtest2.run(strategy=strategy2, initial_amount=1000.0, fees=0.0, delayed_start="2023-05-29")

# combined_results = Results.compare_results([results_random, results_random2])
# print(combined_results.df_statistics.head(10))
# combined_results.ptf_value_plot.show()
# for plot in combined_results.ptf_weights_plot:
#     plot.show()

# print(results_random2.df_statistics.head(10))
# results_random2.ptf_value_plot.show()
# results_random2.ptf_weights_plot.show()
# results_random2.ptf_drawdown_plot.show()