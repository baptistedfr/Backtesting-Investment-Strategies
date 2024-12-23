from my_package.tools import InputType, FrequencyType, Index, Benchmark
from my_package.strategies import *
from my_package import Backtester
from my_package import DataInput
from my_package import Results
import yfinance as yf

'''data = DataInput(data_type=InputType.EQUITY,
                        tickers=['GLE.PA', 'OR.PA','MC.PA','VIV.PA','TTE.PA'],
                        start_date='2015-10-01',
                        end_date='2024-10-01',
                        frequency=FrequencyType.WEEKLY,
                        benchmark=Benchmark.CAC40)'''

data = DataInput(data_type=InputType.CRYPTO,
                 tickers=['BTCUSDT','ETHUSDT','PEPEUSDT','DOGEUSDT','SOLUSDT'],
                 start_date='2018-10-01',
                 end_date='2024-11-15',
                 frequency=FrequencyType.WEEKLY,
                 )


'''# data = DataInput(data_type=InputType.FROM_FILE,
#                         file_path='data/custom.xlsx',
#                         benchmark=Benchmark.CAC40,
#                         )

# print(data.df_prices)


# df = pd.read_excel("data/custom.xlsx")
# print(df)
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
'''

backtest = Backtester(data_input=data)

<<<<<<< HEAD
strategy_momentum = MomentumStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1, is_LS_strategy=True)
strategy_mr = MeanRevertingStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1, is_LS_strategy=True)
strategy_tf = TrendFollowingStrategy(rebalance_frequency=FrequencyType.MONTHLY, short_window_period=10, long_window_period=50, is_LS_strategy=True)
strategy_low = LowVolatilityStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1, is_LS_strategy=True)
strategy_mkw = OptimalSharpeStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1, is_LS_strategy=True)

results_momentum = backtest.run(strategy=strategy_momentum, initial_amount=1000.0, fees=0.0)
results_mr = backtest.run(strategy=strategy_mr, initial_amount=1000.0, fees=0.0)
results_tf = backtest.run(strategy=strategy_tf, initial_amount=1000.0, fees=0.0)
results_low = backtest.run(strategy=strategy_low, initial_amount=1000.0, fees=0.0)
results_mkw = backtest.run(strategy=strategy_mkw, initial_amount=1000.0, fees=0.0)

combined_results = Results.compare_results([results_momentum,results_mr, results_tf, results_low, results_mkw])
=======
# strategy_mr = MeanRevertingStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1)
# results_mr = backtest.run(strategy=strategy_mr, initial_amount=1000.0, fees=0.0)

# # prices = data.df_prices
strategy = RandomFluctuationStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=0)
results_random = backtest.run(strategy=strategy, initial_amount=1000.0, fees=0.000)

# strategy_sharpe = OptimalSharpeStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=0.5)
# strateg_vol = OptimalLowVolatilityStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=0.5)
# results_sharpe = backtest.run(strategy=strategy_sharpe, initial_amount=1000.0, fees=0.0)
# result_vol = backtest.run(strategy=strateg_vol, initial_amount=1000.0, fees=0.0)

strategy_eqw = EqualWeightStrategy(rebalance_frequency = FrequencyType.MONTHLY, lookback_period=0)
results_eqw = backtest.run(strategy=strategy_eqw, initial_amount=1000.0, fees=0.0)

strategie_random_ls = RandomWeightStrategy(rebalance_frequency = FrequencyType.MONTHLY, lookback_period=0)
results_random_ls = backtest.run(strategy=strategie_random_ls, initial_amount=1000.0, fees=0.0)
combined_results = Results.compare_results([results_random,results_eqw,results_random_ls ])

>>>>>>> b554d17c1efe3b485c957be4060a228f60758895
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