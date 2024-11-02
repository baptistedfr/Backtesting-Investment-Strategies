from dataclasses import dataclass, field
from strategy import AbstractStrategy
from typing import Optional
from results import Results
from tools import timer
import pandas as pd
import numpy as np

@dataclass
class Backtester:

    """
    Generic class to backtest strategies from assets prices & a strategy

    Args:
        df_prices (pd.DataFrame) : assets price historic
        initial_amount (float) : initial value of the portfolio
        strategy (AbstractStrategy) : instance of Strategy class with "compute_weights" method
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    df_prices : pd.DataFrame
    initial_amount : float
    strategy : AbstractStrategy
    initial_weights : Optional[list[float]] = None

    ptf_weights : pd.DataFrame = None
    ptf_values : pd.Series = None

    """---------------------------------------------------------------------------------------
    -                               Class computed arguments                                 -
    ---------------------------------------------------------------------------------------"""

    @property
    def df_returns(self) -> pd.DataFrame:
        return self.df_prices.pct_change()
    
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

        """Initialisation"""
        strat_value = self.initial_amount
        weights = self.initial_weights_value
        stored_weights = [weights]
        stored_values = [strat_value]

        for t in range(1, self.backtest_length):
            
            """Compute the portfolio new value"""
            daily_returns = np.array(self.df_returns.iloc[t].values)
            strat_value *= (1 + np.dot(weights, daily_returns))

            """Use Strategy to compute new weights"""
            weights = self.strategy.compute_weights(weights)

            """Store the new computed values"""
            stored_weights.append(weights)
            stored_values.append(strat_value)

        """Prepare the results"""
        self.ptf_weights = pd.DataFrame(stored_weights, index=self.df_returns.index, columns=self.df_returns.columns)
        self.ptf_values = pd.Series(stored_values, index=self.df_returns.index)