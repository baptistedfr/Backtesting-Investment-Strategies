from strategy import AbstractStrategy
from dataclasses import dataclass
from typing import Optional
from results import Results
from tools import timer
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@dataclass
class Backtester:

    """
    Generic class to backtest strategies from assets prices & a strategy

    Args:
        df_prices (pd.DataFrame) : assets price historic
        initial_amount (float) : initial value of the portfolio
        strategy (AbstractStrategy) : instance of Strategy class with "compute_weights" method
        initial_weights (optional list(float)) : initial weights of the strategy, default value is equal weights
        benchmark_prices (optional pd.DataFrame) : benchmark prices to compare the strategy with
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    df_prices : pd.DataFrame
    initial_amount : float

    strategy : AbstractStrategy
    initial_weights : Optional[list[float]] = None

    benchmark_prices : Optional[pd.Series] = None

    ptf_weights : pd.DataFrame = None
    ptf_values : pd.Series = None

    """---------------------------------------------------------------------------------------
    -                               Class computed arguments                                 -
    ---------------------------------------------------------------------------------------"""

    @property
    def df_returns(self) -> pd.DataFrame:
        return self.df_prices.pct_change()
    
    @property
    def benchmark_returns(self) -> pd.DataFrame:
        if self.benchmark_prices is not None:
            return self.benchmark_prices.pct_change()
        else:
            return None
        
    @property
    def backtest_length(self) -> int:
        return len(self.df_returns)
    
    @property
    def nb_assets(self) -> int:
        return self.df_returns.shape[1]
    
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
    def run(self) -> Results :
        """Run the backtest over the asset period (& compare with the benchmark if selected)
        
        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        """Initialisation"""
        strat_value = self.initial_amount
        weights = self.initial_weights_value
        stored_weights = [weights]
        stored_values = [strat_value]

        if self.benchmark_prices is not None :
            benchmark_value = self.initial_amount
            stored_benchmark = [benchmark_value]

        for t in range(1, self.backtest_length):
            
            """Compute the portfolio & benchmark new value"""
            daily_returns = np.array(self.df_returns.iloc[t].values)
            strat_value *= (1 + np.dot(weights, daily_returns))

            """Use Strategy to compute new weights"""
            weights = self.strategy.compute_weights(weights)

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

        return self.output(stored_values, stored_weights, stored_benchmark)
            
            
    def output(self, stored_values : list[float], stored_weights : list[float], stored_benchmark : list[float] = None) -> Results :
        """Create the output for the strategy and its benchmark if selected
        
        Args:
            stored_values (list[float]): Value of the strategy over time
            stored_weights (list[float]): Weights of every asset in the strategy over time
            stored_benchmark (list[float]): Value of the benchmark portfolio over time
        
        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        self.ptf_weights = pd.DataFrame(stored_weights, index=self.df_returns.index, columns=self.df_returns.columns)
        self.ptf_values = pd.Series(stored_values, index=self.df_returns.index)
        results_strat = Results(ptf_values=self.ptf_values, ptf_weights=self.ptf_weights)
        results_strat.get_statistics()
        results_strat.create_plots()

        if self.benchmark_prices is not None :

            benchmark_values = pd.Series(stored_benchmark, index=self.df_returns.index)
            results_bench = Results(ptf_values=benchmark_values)
            results_bench.get_statistics()
            results_bench.create_plots()

            results_strat = results_strat.compare_with(results_bench, name_self="Strategy", name_other="Benchmark")

        return results_strat