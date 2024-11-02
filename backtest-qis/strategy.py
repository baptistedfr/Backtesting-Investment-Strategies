from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class AbstractStrategy(ABC):
    
    @abstractmethod
    def compute_weights(self, previous_weights : np.ndarray[float]) -> np.ndarray[float] :
        """Method used to calculate the new weights of the strategy from given informations"""
        pass

@dataclass
class RandomFluctuationStrategy(AbstractStrategy):
    """Return weights with random fluctuations around the previous weights"""

    def compute_weights(self, previous_weights : np.ndarray[float]) -> np.ndarray[float] :
        new_weights = previous_weights + np.random.random(previous_weights.shape)
        return new_weights / np.sum(new_weights)
    
@dataclass
class FocusedStrategy(AbstractStrategy):
    """
    Strategy fully invested in one asset

    Args:
        asset_index (int) : index of the asset in initial asset list to fully invest in
    """
    asset_index: int

    def compute_weights(self, previous_weights: np.ndarray) -> np.ndarray:
        new_weights = np.zeros_like(previous_weights)
        new_weights[self.asset_index] = 1.0
        return new_weights