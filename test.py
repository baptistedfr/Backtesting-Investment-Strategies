from src.tools import InputType, FrequencyType, Index, Benchmark
from src.strategy import RandomFluctuationStrategy, FocusedStrategy, RandomFluctuationStrategyTEST
from src.backtester import Backtester
from src.data_input import DataInput
from src.results import Results
import pandas as pd

# data = DataInput(data_type=InputType.EQUITY,
#                         tickers=['MC.PA', 'OR.PA'],
#                         start_date='2023-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.DAILY,
#                         benchmark=Benchmark.CAC40)

# data = DataInput(data_type=InputType.CRYPTO,
#                         tickers=['ETHUSDT', 'BTCUSDT', 'SOLUSDT'],
#                         start_date='2023-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.DAILY)
                 
# data = DataInput(data_type=InputType.CUSTOM,
#                         file_path='data/custom.xlsx')

# df = pd.read_excel("data/custom.xlsx")
# data = DataInput(data_type=InputType.FROM_DATAFRAME,
#                 custom_df=df,
#                 benchmark=Benchmark.CAC40,
#                 frequency=FrequencyType.DAILY)

data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                index=Index.CAC40,
                start_date='2023-10-01',
                end_date='2024-03-01',
                frequency=FrequencyType.DAILY,
                benchmark=Benchmark.CAC40)

strategy = RandomFluctuationStrategy()
backtest = Backtester(data_input=data)
results_random = backtest.run(strategy=strategy, initial_amount=1000.0)

strategy2 = RandomFluctuationStrategyTEST()
backtest2 = Backtester(data_input=data)
results_random2 = backtest2.run(strategy=strategy2, initial_amount=1000.0)

combined_results = Results.compare_results([results_random, results_random2])
print(combined_results.df_statistics.head(10))
combined_results.ptf_value_plot.show()
for plot in combined_results.ptf_weights_plot:
    plot.show()

# print(results_random.df_statistics.head(10))
# results_random.ptf_value_plot.show()
# results_random.ptf_weights_plot.show()