from my_package.tools import InputType, FrequencyType, Index, Benchmark
from my_package.strategy import *
from my_package import Backtester
from my_package import DataInput
from my_package import Results

# data = DataInput(data_type=InputType.EQUITY,
#                         tickers=['GLE.PA', 'OR.PA','MC.PA','VIV.PA','TTE.PA'],
#                         start_date='2015-10-01',
#                         end_date='2024-10-01',
#                         frequency=FrequencyType.WEEKLY,
#                         benchmark=Benchmark.CAC40)

# data = DataInput(data_type=InputType.CRYPTO,
#                         tickers=['BTCUSDT','ETHUSDT','PEPEUSDT','DOGEUSDT','SOLUSDT'],
#                         start_date='2018-10-01',
#                         end_date='2024-11-15',
#                         frequency=FrequencyType.WEEKLY,
#                         benchmark = Benchmark.BTC)


# data = DataInput(data_type=InputType.FROM_FILE,
#                         file_path='data/custom.xlsx')

# print(data.df_prices)
# df = pd.read_excel("data/custom.xlsx")
# print(df)
# data = DataInput(data_type=InputType.FROM_DATAFRAME,
#                 custom_df=df,
#                 benchmark=Benchmark.CAC40,
#                 frequency=FrequencyType.DAILY)



data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                index=Index.CAC40,
                start_date='2010-10-01',
                end_date='2024-10-01',
                frequency=FrequencyType.WEEKLY,
                benchmark=Benchmark.CAC40)


backtest = Backtester(data_input=data)

# # prices = data.df_prices
# strategy = RandomFluctuationStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=0)
# backtest = Backtester(data_input=data, custom_name="Fees=0.1%")

#strategy = OptimalSharpeStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=0)


# # # # weight = backtest.initial_weights_value
# # # # print(weight)
#results_random = backtest.run(strategy=strategy, initial_amount=1000.0, fees=0.000)

strategy2 = RandomFluctuationStrategy(rebalance_frequency = FrequencyType.WEEKLY, lookback_period=0)

results_random2 = backtest.run(strategy=strategy2, initial_amount=1000.0, fees=0.0)

strategy3 = EqualWeightStrategy(rebalance_frequency = FrequencyType.MONTHLY, lookback_period=0)
results3 = backtest.run(strategy=strategy3, initial_amount=1000.0, fees=0.0)

combined_results = Results.compare_results([results_random2, results3])
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