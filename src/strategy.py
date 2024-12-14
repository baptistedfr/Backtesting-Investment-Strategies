from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional
from src.tools import FrequencyType


@dataclass
class AbstractStrategy(ABC):
    
    """Abstract class to represent a strategy used in backtesting.
    Args:
        rebalance_frequency : FrequencyType : Choice of the rebalancing frequence of the strategy. Default Value is Monthly rebalancement
        lookback_period : float : The historical period (%year) considered by the strategy to calculate indicators or make decisions. Default value is one year
        adjusted_lookback_period : int : The historical period adjusted by the frequency of the data. 
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    rebalance_frequency : FrequencyType = FrequencyType.MONTHLY 
    lookback_period : float = 1.00 # (1 an) de données  
    adjusted_lookback_period: Optional[int] = None  

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

@dataclass
class TrendFollowingStrategy(AbstractStrategy):
    """Invest in assets that have shown a positive trend"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        mean_returns = historical_data.mean(axis=0)
        positive_trend_assets = mean_returns > 0
        new_weights = positive_trend_assets / np.sum(positive_trend_assets)
        return new_weights


@dataclass
class MomentumStrategy(AbstractStrategy):
    """Invest in assets that have shown positive returns during a recent period"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:

        cumulative_returns = historical_data[-1]
        positive_momentum_assets = cumulative_returns > 0
        new_weights = positive_momentum_assets / np.sum(positive_momentum_assets)
        return new_weights


@dataclass
class LowVolatilityStrategy(AbstractStrategy):
    """Invest in assets with low volatility"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        volatility = historical_data.std(axis=0)

        # If all assets have a volatility of 0, we keep the previous weights
        if np.all(volatility == 0):
            return current_position

        # Inverse of volatility is used to invest more in low volatility assets
        low_volatility_assets = 1 / volatility
        new_weights = low_volatility_assets / np.sum(low_volatility_assets)

        return new_weights


# @dataclass
# class MeanRevertingStrategy(AbstractStrategy):
#     """Invest in assets that have deviated from their historical mean."""
#     "LES PRIX NE SONT PAS STATIONNAIRES PAS DE MOYENNE DE LONG TERME CF MAB"

#     def compute_weights(self, previous_weights: np.ndarray[float], prices: np.ndarray[float]) -> np.ndarray[float]:

#         # Compute the deviation of the current prices from the historical mean
#         mean_prices = prices.mean(axis=0)
#         deviation_from_mean = prices[-1] - mean_prices

#         # Buy assets that are below their historical mean (undervalued)
#         mean_reverting_assets = np.where(deviation_from_mean < 0, -deviation_from_mean, 0)

#         # if all assets are above their historical mean, we do not invest in any asset
#         if np.sum(mean_reverting_assets) == 0:
#             return np.zeros(len(mean_reverting_assets))

#         new_weights = mean_reverting_assets / np.sum(mean_reverting_assets)

#         return new_weights


@dataclass
class RandomFluctuationStrategy(AbstractStrategy):
    """Return weights with random fluctuations around the previous weights"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        new_weights = current_position + np.random.random(current_position.shape) / 4
        new_weights = self.compute_na(new_weights, historical_data[-1])
        return new_weights / np.sum(new_weights)


@dataclass
class FocusedStrategy(AbstractStrategy):
    """
    Strategy fully invested in one asset

    Args:
        asset_index (int): index of the asset in initial asset list to fully invest in
    """
    asset_index: int = 0

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        new_weights = np.zeros_like(current_position)
        new_weights[self.asset_index] = 1.0
        return new_weights
