from src.tools import get_benchmark, InputType, FrequencyType, Index, Benchmark
from src.strategy import RandomFluctuationStrategy, FocusedStrategy
from src.backtester import Backtester
from src.data_input import DataInput

data_equity = DataInput(data_type=InputType.EQUITY,
                        tickers=['MC.PA', 'OR.PA'],
                        start_date='2023-10-01',
                        end_date='2024-10-01',
                        frequency=FrequencyType.DAILY,
                        benchmark=Benchmark.CAC40)

# data_crypto = DataInput(data_type=InputType.CRYPTO,
#                         tickers=['ETHUSDT', 'BTCUSDT', 'SOLUSDT'],
#                         start_date='2023-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.DAILY)
                 
# data_custom = DataInput(data_type=InputType.CUSTOM,
#                         file_path='data/custom.xlsx')

# data_index = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
#                         index=Index.CAC40,
#                         start_date='2023-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.DAILY)

strategy = RandomFluctuationStrategy()
backtest = Backtester(data_input=data_equity, initial_amount=1000.0, strategy=strategy)
results_random = backtest.run()

print(results_random.df_statistics.head(10))
results_random.ptf_value_plot.show()
results_random.ptf_weights_plot.show()

print("end")