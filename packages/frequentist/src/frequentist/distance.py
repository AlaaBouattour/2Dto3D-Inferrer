import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import gaussian_kde

class Distance(ABC):
    @abstractmethod
    def evaluate(self, theta: np.ndarray, data: np.ndarray) -> float:
        """
        Returns a distance (or 'cost') between the empirical data
        and simulated data from the candidate parameter vector theta.
        """
        pass

class KDEDistance(Distance):
    """
    A dimension-agnostic KDE-based distance:
    1) Simulate data under theta using a user-provided simulator.
    2) Build a KDE over the simulated data.
    3) Evaluate the *negative* log-likelihood of observed data under that KDE
    (or some transformation that behaves like a 'distance').
    """
    def __init__(self, simulator, bandwidth=0.2, n_sim=2000):
        self.simulator = simulator
        self.bandwidth = bandwidth
        self.n_sim = n_sim

    def evaluate(self, theta: np.ndarray, data: np.ndarray) -> float:
        self.simulator.set_params(theta)
        sim_data = self.simulator.generate_samples(self.n_sim)
        sim_data_t = sim_data.T
        kde = gaussian_kde(sim_data_t, bw_method=self.bandwidth)
        data_t = data.T
        pdf_vals = kde(data_t)
        pdf_vals[pdf_vals < 1e-300] = 1e-300
        neg_log_like = -np.sum(np.log(pdf_vals))
        return neg_log_like
