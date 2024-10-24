from dataclasses import dataclass
import numpy as np
import random

@dataclass
class Strategy:
    
    def compute_strat(self, previous_weights : np.ndarray[float]) -> np.ndarray[float] :
        new_weights = previous_weights + np.random.random(previous_weights.shape)
        return new_weights / np.sum(new_weights)