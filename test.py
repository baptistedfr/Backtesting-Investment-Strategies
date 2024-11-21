from src.tools import InputType, FrequencyType, Index, Benchmark
from src.strategy import (RandomFluctuationStrategy, FocusedStrategy, TrendFollowingStrategy, MomentumStrategy,
                          LowVolatilityStrategy, MeanRevertingStrategy)
from src.backtester import Backtester
from src.data_input import DataInput
from src.results import Results
import pandas as pd

# data = DataInput(data_type=InputType.EQUITY,
#                         tickers=['GLE.PA', 'OR.PA'],
#                         start_date='2023-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.DAILY,
#                         benchmark=Benchmark.CAC40)

# data = DataInput(data_type=InputType.CRYPTO,
#                         tickers=['BTCUSDT','ETHUSDT','PEPEUSDT','DOGEUSDT','SOLUSDT'],
#                         start_date='2018-10-01',
#                         end_date='2024-11-15',
#                         frequency=FrequencyType.WEEKLY,
#                         benchmark = Benchmark.BTC)


data = DataInput(data_type=InputType.FROM_FILE,
                        file_path='data/custom.xlsx')

# df = pd.read_excel("data/custom.xlsx")
# data = DataInput(data_type=InputType.FROM_DATAFRAME,
#                 custom_df=df,
#                 benchmark=Benchmark.CAC40,
#                 frequency=FrequencyType.DAILY)

# print(data.df_prices)


# data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
#                 index=Index.CAC40,
#                 start_date='2010-10-01',
#                 end_date='2024-10-01',
#                 frequency=FrequencyType.MONTHLY,
#                 benchmark=Benchmark.CAC40)

# prices = data.df_prices
strategy = RandomFluctuationStrategy()
backtest = Backtester(data_input=data, custom_name="Fees=0.1%")
# # weight = backtest.initial_weights_value
# # print(weight)
results_random = backtest.run(strategy=strategy, initial_amount=1000.0, fees=0.001)
print(results_random.df_statistics.head(10))
# strategy2 = RandomFluctuationStrategy()
# backtest2 = Backtester(data_input=data, custom_name="No Fees")
# results_random2 = backtest2.run(strategy=strategy2, initial_amount=1000.0, fees=0.0)

# combined_results = Results.compare_results([results_random, results_random2])
# print(combined_results.df_statistics.head(10))
# combined_results.ptf_value_plot.show()
# for plot in combined_results.ptf_weights_plot:
#     plot.show()

data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                index=Index.CAC40,
                start_date='2023-05-01',
                end_date='2024-11-01',
                frequency=FrequencyType.WEEKLY,
                benchmark=Benchmark.CAC40)

strategy = MeanRevertingStrategy()
backtest = Backtester(data_input=data, custom_name="Mean Reverting - No Fees")
results_random = backtest.run(strategy=strategy, initial_amount=1000.0, fees=0.0, delayed_start="2023-05-29")

strategy2 = LowVolatilityStrategy()
backtest2 = Backtester(data_input=data, custom_name="Low Vol - No Fees")
results_random2 = backtest2.run(strategy=strategy2, initial_amount=1000.0, fees=0.0, delayed_start="2023-05-29")

combined_results = Results.compare_results([results_random, results_random2])
print(combined_results.df_statistics.head(10))
combined_results.ptf_value_plot.show()
for plot in combined_results.ptf_weights_plot:
    plot.show()

# print(results_random.df_statistics.head(10))
results_random.ptf_value_plot.show()
results_random.ptf_weights_plot.show()