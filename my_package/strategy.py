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
        weights[np.any(np.isnan(returns), axis=0)] = 0
        return weights

    def valid_assets_data(self, data):
        """
        Method to filter the data for valid assets (at least one non-NaN value)
        """
        valid_assets = ~np.any(np.isnan(data), axis=0)
        return data[:, valid_assets], valid_assets


@dataclass
class TrendFollowingStrategy(AbstractStrategy):
    """
    Strategy that uses short and long moving averages to determine asset trends.

    Args:
        short_window_period : int : Window size for the short moving average. Can be provided by the user.
        long_window_period : int : Window size for the long moving average. Can be provided by the user.
    """
    short_window_dict = {FrequencyType.DAILY: 10, FrequencyType.WEEKLY: 5, FrequencyType.MONTHLY: 3}
    long_window_dict = {FrequencyType.DAILY: 50, FrequencyType.WEEKLY: 30, FrequencyType.MONTHLY: 12}

    short_window_period: Optional[int] = None
    long_window_period: Optional[int] = None

    @property
    def short_window(self) -> int:
        return self.short_window_period if self.short_window_period is not None else self.short_window_dict[self.rebalance_frequency]

    @property
    def long_window(self) -> int:
        return self.long_window_period if self.long_window_period is not None else self.long_window_dict[self.rebalance_frequency]

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.long_window - 1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        # Filter on valid columns (at least one non-NaN value)
        filtered_data, valid_assets = self.valid_assets_data(data)

        # Calculate the geometric mean of the short & long window periods and adjust it to return both moving averages
        short_MA = gmean(filtered_data[-self.short_window:] + 1) - 1
        long_MA = gmean(filtered_data[-self.long_window:] + 1) - 1

        # Identify assets with a positive trend (short MA greater than long MA)
        signal = short_MA > long_MA

        new_weights = np.zeros(data.shape[1])
        if self.is_LS_strategy:
            # If the strategy is a L/S strategy, we return 1 for a positive trend and -1 for a negative trend
            LS_signal = np.where(signal, 1, -1)
            new_weights[valid_assets] = LS_signal / np.sum(abs(LS_signal))

        # If the strategy is a Long Only strategy
        else:
            # If no assets have a positive trend, return an array of zeros
            if np.sum(signal) == 0:
                return new_weights
            else:
                new_weights[valid_assets] = signal / np.sum(signal)

        return new_weights


@dataclass
class MomentumStrategy(AbstractStrategy):
    """Invest in assets that have shown positive returns during a recent period."""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period-1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        # Filter on valid columns (at least one non-NaN value)
        filtered_data, valid_assets = self.valid_assets_data(data)

        # Calculate the geometric mean of the data and adjust it to return the mean return
        mean_return = gmean(filtered_data + 1) - 1

        # Identify assets with positive momentum (mean return greater than 0)
        signal = mean_return > 0

        new_weights = np.zeros(data.shape[1])
        if self.is_LS_strategy:
            # If the strategy is a L/S strategy, we return 1 for a positive momentum and -1 for a negative momentum
            new_weights[valid_assets] = mean_return / abs(np.sum(mean_return))

        # If the strategy is a Long Only strategy
        else:
            # If no assets have positive momentum, return an array of zeros
            if np.sum(signal) == 0:
                return new_weights
            else:
                new_weights[valid_assets] = (mean_return * signal) / np.sum(mean_return * signal)

        return new_weights


@dataclass
class LowVolatilityStrategy(AbstractStrategy):
    """Invest in assets with low volatility"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period - 1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        # Filter on valid columns (at least one non-NaN value)
        filtered_data, valid_assets = self.valid_assets_data(data)

        # Calculate the standard deviation of the returns
        volatility = filtered_data.std(axis=0)

        # Inverse of volatility is used to invest more in low volatility assets
        low_volatility_assets = 1 / volatility

        new_weights = np.zeros(data.shape[1])
        new_weights[valid_assets] = low_volatility_assets / np.sum(low_volatility_assets)

        return new_weights


@dataclass
class MeanRevertingStrategy(AbstractStrategy):
    """Invest in assets that have deviated from their historical mean."""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period - 1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
        # Filter on valid columns (at least one non-NaN value)
        filtered_data, valid_assets = self.valid_assets_data(data)

        # Calculate the deviation of the latest data point from the mean of the data, and the standard deviation of the data
        deviation = filtered_data[-1] - np.mean(filtered_data[:-1], axis=0)
        std_prices = np.std(filtered_data[:-1], axis=0)

        # Calculate the z-scores for the data
        z_scores = deviation / std_prices

        # Identify undervalued assets (those with negative z-scores)
        signal = z_scores < 0

        new_weights = np.zeros(data.shape[1])
        if self.is_LS_strategy:
            new_weights[valid_assets] = -z_scores / abs(np.sum(z_scores))

        # If the strategy is a Long Only strategy
        else:
            # If there are no undervalued assets, return an array of zeros
            if np.sum(signal) == 0:
                return new_weights
            else:
                new_weights[valid_assets] = (-z_scores * signal) / np.sum(-z_scores * signal)

        return new_weights

@dataclass
class OptimalSharpeStrategy(AbstractStrategy):
    """Invest in assets that maximizes the Sharpe Ratio calculated with Markowitz optimization"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period-1:]
        new_weights = self.fit(data)
        return new_weights
    
    def fit(self, data: np.ndarray[float]):
        # Filter on valid columns (at least one non-NaN value)
        filtered_data, valid_assets = self.valid_assets_data(data)
        
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
