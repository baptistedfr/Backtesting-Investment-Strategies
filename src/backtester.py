from src.strategy import AbstractStrategy
from src.data_input import DataInput
from dataclasses import dataclass
from src.results import Results
from datetime import datetime
from src.tools import timer, FrequencyType
from typing import Optional
import pandas as pd
import numpy as np
from functools import cached_property

@dataclass
class Backtester:

    """
    Generic class to backtest strategies from assets prices & a strategy

    Args:
        data_input (DataInput) : data input object containing assets prices historic
        initial_weights (optional list(float)) : initial weights of the strategy, default value is equal weights
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    data_input : DataInput
    custom_name : str = None

    ptf_weights : pd.DataFrame = None
    ptf_values : pd.Series = None

    """---------------------------------------------------------------------------------------
    -                               Class computed arguments                                 -
    ---------------------------------------------------------------------------------------"""

    @cached_property
    def df_prices(self) -> pd.DataFrame:
        return self.data_input.df_prices
    
    @cached_property
    def dates(self) -> list[datetime]:
        return self.data_input.df_prices["Date"]
    
    @cached_property
    def df_returns(self) -> pd.DataFrame:
        return self.df_prices.iloc[:, 1:].pct_change()
    
    @cached_property
    def benchmark_prices(self) -> pd.Series:
        if self.data_input.benchmark is not None:
            return self.data_input.df_benchmark
        else:
            return None
        
    @cached_property
    def benchmark_returns(self) -> pd.Series:
        if self.benchmark_prices is not None:
            return self.benchmark_prices.iloc[:, 1].pct_change()
        else:
            return None
        
    @cached_property
    def backtest_length(self) -> int:
        return len(self.df_prices)
    
    @cached_property
    def nb_assets(self) -> int:
        return self.df_prices.shape[1]-1
    
    @cached_property
    def initial_weights_value(self) -> np.ndarray:
        """
        Initialisation des poids initiaux
        """
        initial_prices = self.df_prices.iloc[0,1:]
        if self.data_input.initial_weights is None:
            weights = np.full(self.nb_assets, 1 / self.nb_assets)
            weights[initial_prices.isna()] = 0
            
        else:
            weights = np.array(self.data_input.initial_weights)
            if len(weights) != self.nb_assets:
                raise ValueError("The initial weights size must match the number of assets.")
            weights[initial_prices.isna()] = 0

        weights /= weights.sum() # Normalisation des poids
        return weights

    """---------------------------------------------------------------------------------------
    -                                   Class methods                                        -
    ---------------------------------------------------------------------------------------"""

    @timer
    def run(self, strategy : AbstractStrategy, initial_amount : float = 1000.0, fees : float = 0.001, delayed_start : str = None) -> Results :
        """Run the backtest over the asset period (& compare with the benchmark if selected)
        
        Args:
            strategy (AbstractStrategy) : instance of Strategy class with "compute_weights" method
            initial_amount (float) : initial value of the portfolio
            fees (float) : transaction fees for every portfolio weight rebalancing
            delayed_start (optional str) : possibility to start the backtest after the first date of the data input 
                                           (used in the strategies to have enough data at the beginning of the backtest)

        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """
        
        """Verification of some basic instances"""
        if not isinstance(initial_amount, (int, float)) or initial_amount <= 0:
            raise ValueError("Initial amount must be a positive number.")
        if not (0 <= fees <= 1):
            raise ValueError("Fees must be a proportion between 0 and 1.")
        
        if strategy.rebalance_frequency.value > self.data_input.frequency.value:
            raise ValueError("We cannot have a frequency of rebalancement more frequent than the frequency of the prices !")
        
        """Get the rebalancing dates"""
        rebalancing_dates = self._get_rebalancing_dates(strategy.rebalance_frequency)

        """Initialisation"""
        strat_value = initial_amount
        returns_matrix = self.df_returns.to_numpy()
        weights = self.initial_weights_value
        stored_weights = [weights]
        stored_values = [strat_value]
        stored_benchmark = None
        benchmark_returns_matrix = self.benchmark_returns
        if benchmark_returns_matrix is not None :
            benchmark_value = initial_amount
            stored_benchmark = [benchmark_value]
            benchmark_returns_matrix = benchmark_returns_matrix.to_numpy()

        if delayed_start is not None:
            self.start_backtest = self.dates[self.dates == delayed_start].index[0]
        else :
            self.start_backtest = 1

        for shift, t in enumerate(range(self.start_backtest + 1, self.backtest_length)):
            
            """Compute the portfolio & benchmark new value"""
            daily_returns = np.nan_to_num(returns_matrix[t], nan=0.0)
            new_strat_value = strat_value * (1 + np.dot(weights, daily_returns))

            if t in rebalancing_dates:
                """Use Strategy to compute new weights (Rebalancement)"""
                new_weights = strategy.compute_weights(weights, returns_matrix[shift+1:t])
                """Compute transaction costs"""
                transaction_costs = strat_value * fees * np.sum(np.abs(new_weights - weights))
                new_strat_value -= transaction_costs
            else: 
                """Apply drift to weights"""
                drifted_weights = weights * (1 + daily_returns)
                new_weights = drifted_weights / drifted_weights.sum()
                
            """Store the new computed values"""
            stored_weights.append(new_weights)
            stored_values.append(new_strat_value)

            """Compute & sotre the new benchmark value"""
            if self.benchmark_prices is not None :
                benchmark_rdt = benchmark_returns_matrix[t]
                benchmark_value *= (1 + benchmark_rdt)
                stored_benchmark.append(benchmark_value)

            weights = new_weights
            strat_value = new_strat_value
        strat_name = self.custom_name if self.custom_name is not None else strategy.__class__.__name__
        return self.output(strat_name, stored_values, stored_weights, stored_benchmark)
            
    @timer
    def output(self, strategy_name : str, stored_values : list[float], stored_weights : list[float], stored_benchmark : list[float] = None) -> Results :
        """Create the output for the strategy and its benchmark if selected
        
        Args:
            stored_values (list[float]): Value of the strategy over time
            stored_weights (list[float]): Weights of every asset in the strategy over time
            stored_benchmark (list[float]): Value of the benchmark portfolio over time
            strategy_name (str) : Name of the current strategy

        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        self.ptf_weights = pd.DataFrame(stored_weights, index=self.dates[self.start_backtest:], columns=self.df_returns.columns)
        self.ptf_values = pd.Series(stored_values, index=self.dates[self.start_backtest:])
        results_strat = Results(ptf_values=self.ptf_values, ptf_weights=self.ptf_weights, strategy_name=strategy_name, data_frequency=self.data_input.frequency)
        results_strat.get_statistics()
        results_strat.create_plots()

        if stored_benchmark is not None :

            benchmark_values = pd.Series(stored_benchmark, index=self.dates[self.start_backtest:])
            results_bench = Results(ptf_values=benchmark_values, strategy_name="Benchmark", data_frequency=self.data_input.frequency)
            results_bench.get_statistics()
            results_bench.create_plots()
            results_strat = Results.compare_results([results_strat, results_bench])

        return results_strat
    
    def _get_rebalancing_dates(self, frequency: FrequencyType) -> list[int]:
        """
        Repère les indices correspondant aux dates de rebalancement en fonction de la fréquence donnée.
        
        Args:
            frequency (FrequencyType): Fréquence de rebalancement souhaitée.
        
        Returns:
            list[int]: Indices des dates de rebalancement.
        """
        if not pd.api.types.is_datetime64_any_dtype(self.dates):
            self.data_input.df_prices["Date"] = pd.to_datetime(self.data_input.df_prices["Date"])

        rebalancing_dates = []
        if frequency == FrequencyType.MONTHLY:
            rebalancing_dates = self.data_input.df_prices.groupby(self.df_prices["Date"].dt.to_period("M")).apply(
                lambda group: group.index[-1]).tolist()
        elif frequency == FrequencyType.WEEKLY:
            rebalancing_dates = self.df_prices.groupby(self.df_prices["Date"].dt.to_period("W")).apply(
                lambda group: group.index[-1]).tolist()
        elif frequency == FrequencyType.DAILY:
            rebalancing_dates = list(range(len(self.dates)))  # Toutes les dates sont incluses
        
        return rebalancing_dates