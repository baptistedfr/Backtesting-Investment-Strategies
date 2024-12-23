import numpy as np
from dataclasses import dataclass
from .abstract_strategy import AbstractStrategy


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
        return 
    
@dataclass
class RandomWeightStrategy(AbstractStrategy):
    """Return random weights with Long Short"""

    def get_position(self, historical_data : np.ndarray[float], current_position: np.ndarray[float]) -> np.ndarray[float]:
        random_weights = np.random.uniform(-1, 1, current_position.shape) #Generate random weights between -1 and 1
        random_weights = self.compute_na(random_weights, historical_data[-1])
        #Adjust weights to ensure the sum is 0
        mean_adjustment = np.mean(random_weights)
        adjusted_weights = random_weights - mean_adjustment
        new_weights = (adjusted_weights / np.sum(np.abs(adjusted_weights)))
        return new_weights
    
