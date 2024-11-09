from src.strategy import AbstractStrategy
from src.data_input import DataInput
from dataclasses import dataclass
from src.results import Results
from datetime import datetime
from src.tools import timer
from typing import Optional
import pandas as pd
import numpy as np

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
    initial_weights : Optional[list[float]] = None

    ptf_weights : pd.DataFrame = None
    ptf_values : pd.Series = None

    """---------------------------------------------------------------------------------------
    -                               Class computed arguments                                 -
    ---------------------------------------------------------------------------------------"""

    @property
    def df_prices(self) -> pd.DataFrame:
        return self.data_input.df_prices
    
    @property
    def dates(self) -> list[datetime]:
        return self.df_prices["Date"]
    @property
    def df_returns(self) -> pd.DataFrame:
        return self.df_prices.iloc[:, 1:].pct_change()
    
    @property
    def benchmark_prices(self) -> pd.Series:
        if self.data_input.benchmark is not None:
            return self.data_input.df_benchmark
        else:
            return None
        
    @property
    def benchmark_returns(self) -> pd.Series:
        if self.benchmark_prices is not None:
            return self.benchmark_prices.pct_change()
        else:
            return None
        
    @property
    def backtest_length(self) -> int:
        return len(self.df_prices)
    
    @property
    def nb_assets(self) -> int:
        return self.df_prices.shape[1] - 1
    
    @property
    def initial_weights_value(self) -> np.ndarray:
        if self.initial_weights is None:
            return np.full(self.nb_assets, 1 / self.nb_assets)
        else:
            return self.initial_weights

    """---------------------------------------------------------------------------------------
    -                                   Class methods                                        -
    ---------------------------------------------------------------------------------------"""

    @timer
    def run(self, strategy : AbstractStrategy, initial_amount : float = 1000.0) -> Results :
        """Run the backtest over the asset period (& compare with the benchmark if selected)
        
        Args:
            strategy (AbstractStrategy) : instance of Strategy class with "compute_weights" method
            initial_amount (float) : initial value of the portfolio
        
        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        """Initialisation"""
        strat_value = initial_amount
        weights = self.initial_weights_value
        stored_weights = [weights]
        stored_values = [strat_value]
        returns_matrix = self.df_returns.to_numpy()

        if self.benchmark_prices is not None :
            benchmark_value = initial_amount
            stored_benchmark = [benchmark_value]

        for t in range(1, self.backtest_length):
            
            """Compute the portfolio & benchmark new value"""
            daily_returns = returns_matrix[t]
            strat_value *= (1 + np.dot(weights, daily_returns))

            """Use Strategy to compute new weights"""
            weights = strategy.compute_weights(weights)

            """Store the new computed values"""
            stored_weights.append(weights)
            stored_values.append(strat_value)

            """Compute & sotre the new benchmark value"""
            if self.benchmark_prices is not None :
                benchmark_rdt = self.benchmark_returns.iloc[t]
                benchmark_value *= (1 + benchmark_rdt)
                stored_benchmark.append(benchmark_value)

        if self.benchmark_prices is None :
            stored_benchmark = None

        return self.output(strategy.__class__.__name__, stored_values, stored_weights, stored_benchmark)
            
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

        self.ptf_weights = pd.DataFrame(stored_weights, index=self.dates, columns=self.df_returns.columns)
        self.ptf_values = pd.Series(stored_values, index=self.dates)
        results_strat = Results(ptf_values=self.ptf_values, ptf_weights=self.ptf_weights, strategy_name=strategy_name)
        results_strat.get_statistics()
        results_strat.create_plots()

        if self.benchmark_prices is not None :

            benchmark_values = pd.Series(stored_benchmark, index=self.dates)
            results_bench = Results(ptf_values=benchmark_values, strategy_name="Benchmark")
            results_bench.get_statistics()
            results_bench.create_plots()

            # results_strat = results_strat.compare_with(results_bench, name_self=strategy_name, name_other="Benchmark")
            results_strat = Results.compare_results([results_strat, results_bench])

        return results_strat