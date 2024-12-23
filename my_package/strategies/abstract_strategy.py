from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional

from ..tools import FrequencyType

@dataclass
class AbstractStrategy(ABC):
    
    """Abstract class to represent a strategy used in backtesting.
    Args:
        rebalance_frequency : FrequencyType : Choice of the rebalancing frequence of the strategy. Default Value is Monthly rebalancement
        lookback_period : float : The historical period (%year) considered by the strategy to calculate indicators or make decisions. Default value is one year
        adjusted_lookback_period : int : The lookback_period adjusted by the frequency of the data. Automatically calculated in the Backtester
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    rebalance_frequency: FrequencyType = FrequencyType.MONTHLY
    lookback_period: float = 1.00  # 1 year of data
    adjusted_lookback_period: Optional[int] = None
    is_LS_strategy: Optional[bool] = False

    """---------------------------------------------------------------------------------------
    -                                 Class methods                                       -
    ---------------------------------------------------------------------------------------"""

    @abstractmethod
    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Mandatory method to be implemented by all strategies.
        Calculates the new position based on historical data and the current position.

        Args:
            historical_data : Historical data required for decision-making (e.g., prices, returns, etc.).
            current_position : The current position of the strategy (e.g., current asset weights).
        
        Returns:
            The new positions in a numpy array
        """
        pass

    def fit(self, data):
        """
        Optional method.
        Can be used to train or calibrate the strategy (does nothing by default).
        
        Args:
            data : Data required to train the strategy.
        """
        pass

    def compute_na(self, weights, returns):
        """
        Method to handle NaN values by setting weights to 0 where returns are NaN and adjusts the weights correspondously
        """
        weights[np.isnan(returns)] = 0
        return weights

    def valid_assets_data(self, data):
        """
        Method to filter the data for valid assets (at least one non-NaN value)
        """
        valid_assets = ~np.any(np.isnan(data), axis=0)
        return data[:, valid_assets], valid_assets
