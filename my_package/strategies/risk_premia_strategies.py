from scipy.stats import gmean
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
<<<<<<< HEAD:my_package/strategy.py
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
=======
from dataclasses import dataclass
from .abstract_strategy import AbstractStrategy
from ..tools import FrequencyType
from typing import Optional

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



>>>>>>> b554d17c1efe3b485c957be4060a228f60758895:my_package/strategies/risk_premia_strategies.py


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

<<<<<<< HEAD:my_package/strategy.py
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


=======
    def fit(self, data:np.ndarray[float]):
        mean_return = gmean(data + 1) - 1
        positive_momentum_assets = mean_return > 0
        if np.sum(positive_momentum_assets) == 0:
            return np.zeros(len(positive_momentum_assets))
        return data * positive_momentum_assets / np.sum(data[positive_momentum_assets])
    
>>>>>>> b554d17c1efe3b485c957be4060a228f60758895:my_package/strategies/risk_premia_strategies.py
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


'''@dataclass
class LowVolatilityStrategy(AbstractStrategy):
    """Invest in assets with low volatility"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        data = historical_data[-self.adjusted_lookback_period - 1:]
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray[float]):
<<<<<<< HEAD:my_package/strategy.py
        # Filter on valid columns (at least one non-NaN value)
        filtered_data, valid_assets = self.valid_assets_data(data)
        
        expected_returns = np.nanmean(filtered_data, axis=0)
        cov_matrix = np.cov(filtered_data, rowvar=False)
=======
        volatility = data.std(axis=0)

        # If all assets have a volatility of 0, we keep the previous weights
        #if np.all(volatility == 0):
        #    return current_position

        # Inverse of volatility is used to invest more in low volatility assets
        low_volatility_assets = 1 / volatility

        return low_volatility_assets / np.sum(low_volatility_assets)'''
>>>>>>> b554d17c1efe3b485c957be4060a228f60758895:my_package/strategies/risk_premia_strategies.py





<<<<<<< HEAD:my_package/strategy.py
@dataclass
class RandomFluctuationStrategy(AbstractStrategy):
    """Return weights with random fluctuations around the previous weights"""
=======
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


>>>>>>> b554d17c1efe3b485c957be4060a228f60758895:my_package/strategies/risk_premia_strategies.py


