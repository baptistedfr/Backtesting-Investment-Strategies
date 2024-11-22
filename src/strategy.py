from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class AbstractStrategy(ABC):

    @abstractmethod
    def compute_weights(self, previous_weights: np.ndarray[float], returns: np.ndarray[float]) -> np.ndarray[float]:
        """Method used to calculate the new weights of the strategy from given information"""
        pass

    def compute_na(self, weights, returns):
        weights[np.isnan(returns)] = 0
        return weights

@dataclass
class TrendFollowingStrategy(AbstractStrategy):
    """Invest in assets that have shown a positive trend"""

    def compute_weights(self, previous_weights: np.ndarray[float], returns: np.ndarray[float]) -> np.ndarray[float]:
        mean_returns = returns.mean(axis=0)
        positive_trend_assets = mean_returns > 0
        new_weights = positive_trend_assets / np.sum(positive_trend_assets)
        return new_weights


'''@dataclass
class ValueStrategy(AbstractStrategy):
    """
    Invest in undervalued assets.
    """

    def compute_weights(self, previous_weights: np.ndarray, df_prices: pd.DataFrame) -> np.ndarray:
        # to complete by getting the PE ratios
        new_weights = undervalued_assets / np.sum(undervalued_assets)
        return new_weights'''


@dataclass
class MomentumStrategy(AbstractStrategy):
    """Invest in assets that have shown positive returns during a recent period"""

    def compute_weights(self, previous_weights: np.ndarray[float], returns: np.ndarray[float]) -> np.ndarray[float]:

        cumulative_returns = returns[-1]
        positive_momentum_assets = cumulative_returns > 0
        new_weights = positive_momentum_assets / np.sum(positive_momentum_assets)
        return new_weights


@dataclass
class LowVolatilityStrategy(AbstractStrategy):
    """Invest in assets with low volatility"""

    def compute_weights(self, previous_weights: np.ndarray[float], returns: np.ndarray[float]) -> np.ndarray[float]:
        volatility = returns.std(axis=0)

        # If all assets have a volatility of 0, we keep the previous weights
        if np.all(volatility == 0):
            return previous_weights

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

    def compute_weights(self, previous_weights: np.ndarray[float], returns) -> np.ndarray[float]:
        new_weights = previous_weights + np.random.random(previous_weights.shape) / 4
        new_weights = self.compute_na(new_weights, returns[-1])
        return new_weights / np.sum(new_weights)


@dataclass
class FocusedStrategy(AbstractStrategy):
    """
    Strategy fully invested in one asset

    Args:
        asset_index (int): index of the asset in initial asset list to fully invest in
    """
    asset_index: int = 0

    def compute_weights(self, previous_weights: np.ndarray[float]) -> np.ndarray[float]:
        new_weights = np.zeros_like(previous_weights)
        new_weights[self.asset_index] = 1.0
        return new_weights
