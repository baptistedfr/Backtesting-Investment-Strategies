import numpy as np
from dataclasses import dataclass
from .abstract_strategy import AbstractStrategy

@dataclass
class KernelSkewStrategy(AbstractStrategy):
    """
    Strategy based on Non-Parametric Density Estimation by Kernel Density Estimation (KDE) method.
    To simplify the process we will use an estimation of the bandwith parameter 'h' given by Silverman.
    
    The strategy is based on the estimation of the skew of the kernel density (analytic calculus can be derived from the KDE formula).
    """
    def silverman_bandwidth(self, data: np.ndarray) -> float:
        """Compute the bandwidth using Silverman's rule of thumb."""
        sigma = np.std(data)
        n = len(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bandwidth = 0.9 * min(sigma, iqr / 1.34) * n**(-0.2)
        return bandwidth

    def calculate_skew(self, returns: np.ndarray, h: float) -> float:
        """Calculate the skewness of the kernel density estimate."""
        r3 = np.mean(returns**3)
        r2 = np.mean(returns**2)
        mu = np.mean(returns)
        std_h = (np.var(returns) + h**2)**1.5
        if np.isfinite(std_h) and std_h > 0:
            skewness = (r3 + 3 * mu * h**2 - 3 * mu * (r2 + h**2) + mu**3) / std_h
        else:
            skewness = 0.0
        return skewness

    def get_position(self, historical_data: np.ndarray, current_position: np.ndarray) -> np.ndarray:
        """
        Compute new portfolio weights based on KDE skewness.
        
        Args:
            historical_data (np.ndarray): Historical price data (time x assets).
            current_position (np.ndarray): Current portfolio weights.
        
        Returns:
            np.ndarray: New portfolio weights.
        """
        historical_data = historical_data[-self.adjusted_lookback_period-1:]
        
        new_weights = self.fit(historical_data, current_position)
        
        return new_weights
    
    def fit(self, historical_data: np.ndarray, current_position: np.ndarray):
        

        returns = np.diff(historical_data, axis=0) / historical_data[:-1]  # Calculate daily returns
        num_assets = returns.shape[1]
        
        skews = []
        for i in range(num_assets):
            asset_returns = returns[:, i]
            h = self.silverman_bandwidth(asset_returns)
            skewness = self.calculate_skew(asset_returns, h)
            skews.append(skewness)
        
        # Select assets with the lowest skewness (indicating asymmetry)
        num_selected = int(np.sum(current_position > 0)) or 1  # Ensure at least 1 asset is selected
        selected_indices = np.argsort(skews)[:num_selected]
        
        # Allocate weights equally among selected assets
        new_weights = np.zeros(num_assets)
        new_weights[selected_indices] = 1.0 / num_selected

        return new_weights