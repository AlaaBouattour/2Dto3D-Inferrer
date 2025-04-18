import numpy as np
from abc import ABC, abstractmethod

class Prior(ABC):
    @abstractmethod
    def log_prior(self, theta: np.ndarray) -> float:
        """
        Evaluate log p(theta). Must return -inf if invalid.
        """
        pass


class Uniform(Prior):
    """
    An N-dimensional uniform prior, optionally enforcing ascending order
    across all dimensions: theta[0] < theta[1] < ... < theta[n-1].
    """
    def __init__(self, bounds, enforce_order=False):
        """
        Parameters
        ----------
        bounds : list of (low, high)
            Each tuple defines the valid range for one dimension.
        enforce_order : bool
            If True, requires theta[i] < theta[i+1] for all i.
        """
        self.bounds = bounds
        self.enforce_order = enforce_order

    def log_prior(self, theta: np.ndarray) -> float:
        # 1) Check dimension matches
        if len(theta) != len(self.bounds):
            return -np.inf

        # 2) Check each component is in [low, high]
        for val, (low, high) in zip(theta, self.bounds):
            if val < low or val > high:
                return -np.inf

        # 3) If enforce_order, check ascending for all consecutive pairs
        if self.enforce_order:
            for i in range(len(theta) - 1):
                if not (theta[i] < theta[i+1]):
                    return -np.inf

        # Uniform => log-likelihood is constant (return 0, ignoring normalization)
        return 0.0
