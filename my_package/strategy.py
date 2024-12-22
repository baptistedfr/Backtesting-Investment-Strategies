from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from .tools import FrequencyType
from scipy.stats import gmean

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
class ValueStrategy(AbstractStrategy):

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        per = historical_data[-1]
        new_weights = self.fit(per)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        undervalued_assets = data < 10
        #overvalued_assets = data > 17
        return data * undervalued_assets / np.sum(data[undervalued_assets])





@dataclass
class TrendFollowingStrategy(AbstractStrategy):
    """
    Strategy that uses short and long moving averages to determine asset trends.

    Args:
        short_window : int : Window size for the short moving average. Must be provided by the user.
        long_window : int : Window size for the long moving average. Must be provided by the user.
    """
    short_window_dict = {FrequencyType.DAILY: 10, FrequencyType.WEEKLY: 5, FrequencyType.MONTHLY: 3}
    long_window_dict = {FrequencyType.DAILY: 50, FrequencyType.WEEKLY: 30, FrequencyType.MONTHLY: 12}

    short_window_user: Optional[int] = None
    long_window_user: Optional[int] = None

    @property
    def short_window(self) -> int:
        return self.short_window_user if self.short_window_user is not None else self.short_window_dict[self.rebalance_frequency]

    @property
    def long_window(self) -> int:
        return self.long_window_user if self.long_window_user is not None else self.long_window_dict[self.rebalance_frequency]

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        new_weights = self.fit(historical_data)
        return new_weights

    def fit(self, data:np.ndarray[float]):
        short_MA = gmean(data[-self.short_window:] + 1) - 1
        long_MA = gmean(data[-self.long_window:] + 1) - 1
        positive_trend_assets = short_MA > long_MA
        if np.sum(positive_trend_assets) == 0:
            return np.zeros(len(positive_trend_assets))
        return positive_trend_assets / np.sum(positive_trend_assets)


@dataclass
class MomentumStrategy(AbstractStrategy):
    """Strategy that invests in assets that have shown positive returns during a recent period."""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period-1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data:np.ndarray[float]):
        mean_return = gmean(data + 1) - 1
        positive_momentum_assets = mean_return > 0
        if np.sum(positive_momentum_assets) == 0:
            return np.zeros(len(positive_momentum_assets))
        return data * positive_momentum_assets / np.sum(data[positive_momentum_assets])


'''@dataclass
class LowVolatilityStrategy(AbstractStrategy):
    """Invest in assets with low volatility"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period - 1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        volatility = data.std(axis=0)

        # If all assets have a volatility of 0, we keep the previous weights
        #if np.all(volatility == 0):
        #    return current_position

        # Inverse of volatility is used to invest more in low volatility assets
        low_volatility_assets = 1 / volatility

        return low_volatility_assets / np.sum(low_volatility_assets)'''


@dataclass
class MeanRevertingStrategy(AbstractStrategy):
    """Invest in assets that have deviated from their historical mean."""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period - 1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        # Calculate the deviation of the latest data point from the mean of the data, and the standard deviation of the data
        deviation = data[-1] - np.mean(data, axis=0)
        std_prices = np.std(data, axis=0)

        # Calculate the z-scores for the data
        z_scores = deviation / std_prices

        # Identify undervalued assets (those with negative z-scores)
        undervalued_assets = z_scores < 0

        # If there are no undervalued assets, return an array of zeros
        if np.sum(undervalued_assets) == 0:
            return np.zeros(len(undervalued_assets))

        # Calculate the new weights for the undervalued assets
        return -z_scores * undervalued_assets / np.sum(z_scores[undervalued_assets])

@dataclass
class OptimalSharpeStrategy(AbstractStrategy):
    """Invest in assets that maximizes the Sharpe Ratio calculated with Markowitz optimization"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period-1:]
        new_weights = self.fit(data)
        return new_weights
    
    def fit(self, data: np.ndarray[float]):
        # Identify valid columns (at least one non-NaN value)
        valid_assets = ~np.any(np.isnan(data), axis=0)
        # Filter the data for valid assets
        filtered_data = data[:, valid_assets]
        
        expected_returns = np.nanmean(filtered_data, axis=0)
        cov_matrix = np.cov(filtered_data, rowvar=False)

        n_assets = len(expected_returns)
        x0 = np.ones(n_assets) / n_assets

        bounds = Bounds(0, 1)
        linear_constraint = LinearConstraint(np.ones((n_assets,), dtype=int), 1, 1)  # Sum of weights = 1

        def max_sharpe(w):
            return -expected_returns.dot(w) / np.sqrt(w.T.dot(cov_matrix).dot(w))
        
        # Perform optimization
        result = minimize(max_sharpe, x0, method='trust-constr', constraints=linear_constraint, bounds=bounds)
        # Rebuild the full weight array, assigning 0 to invalid assets
        full_weights = np.zeros(data.shape[1])
        full_weights[valid_assets] = result.x
        return full_weights
    
@dataclass
class OptimalLowVolatilityStrategy(AbstractStrategy):
    """Invest in assets that minimizes the Volatility calculated with markovitz optimization"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period-1:]
        new_weights = self.fit(data)
        return new_weights
    
    def fit(self, data:np.ndarray[float]):
        # Identify valid columns (at least one non-NaN value)
        valid_assets = ~np.any(np.isnan(data), axis=0)
        # Filter the data for valid assets
        filtered_data = data[:, valid_assets]
        
        expected_returns = np.nanmean(filtered_data, axis=0)
        cov_matrix = np.cov(filtered_data, rowvar=False)

        n_assets = len(expected_returns)
        x0 = np.ones(n_assets) / n_assets

        bounds = Bounds(0, 1)
        linear_constraint = LinearConstraint(np.ones((n_assets,), dtype=int), 1, 1)  # Sum of weights = 1

        def min_vol(w):
            return np.sqrt(w.T.dot(cov_matrix).dot(w))
        
        # Perform optimization
        result = minimize(min_vol, x0, method='trust-constr', constraints=linear_constraint, bounds=bounds)
        # Rebuild the full weight array, assigning 0 to invalid assets
        full_weights = np.zeros(data.shape[1])
        full_weights[valid_assets] = result.x
        return full_weights


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
class EqualWeightStrategy(AbstractStrategy):
    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Allocates equal weights to all assets in the portfolio
        """
        n_assets = historical_data.shape[1]
        new_weights = np.ones(n_assets) / n_assets
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
