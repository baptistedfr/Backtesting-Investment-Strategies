import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from dataclasses import dataclass
from .abstract_strategy import AbstractStrategy

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
