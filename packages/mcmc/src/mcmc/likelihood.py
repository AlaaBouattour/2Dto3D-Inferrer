import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import gaussian_kde, entropy

class Likelihood(ABC):
    @abstractmethod
    def log_likelihood(self, theta: np.ndarray, data: np.ndarray) -> float:
        """
        Evaluate log p(data | theta). Return -inf if invalid.
        """
        pass


class KDE(Likelihood):
    """
    A dimension-agnostic KDE-based likelihood.
    - Expects data of shape (N, d).
    - Builds a d-dimensional KDE over simulated data of shape (M, d).
    """
    def __init__(self, simulator, param_names, bandwidth=0.1, n_sim=2000):
        """
        simulator : object with set_params(list) and generate_samples(N)->(N,d)
        param_names : list of strings matching the dimension of theta
        bandwidth : float or str, bandwidth for gaussian_kde
        n_sim : number of simulated samples
        """
        self.simulator = simulator
        self.param_names = param_names
        self.bandwidth = bandwidth
        self.n_sim = n_sim

    def log_likelihood(self, theta: np.ndarray, data: np.ndarray) -> float:
        self.simulator.set_params(theta)

        # 1) Forward simulate
        sim_data = self.simulator.generate_samples(self.n_sim)
        # sim_data should be shape (self.n_sim, d)

        # 2) Build d-dimensional KDE
        # scikit's gaussian_kde expects shape (d, N_samples).
        sim_data_t = sim_data.T  # shape (d, self.n_sim)
        kde = gaussian_kde(sim_data_t, bw_method=self.bandwidth)

        # 3) Evaluate log-likelihood of observed data (N, d)
        data_t = data.T
        pdf_vals = kde(data_t)  # shape (N,)
        # Eviter -inf dans le log
        pdf_vals[pdf_vals < 1e-300] = 1e-300

        return np.sum(np.log(pdf_vals))


class KL(Likelihood):
    """
    A dimension-agnostic KL-based likelihood.
    - For data of shape (N, d), we do a 1D histogram for each dimension (column).
    - Sim data is also shape (M, d). We do a 1D histogram for each dimension.
    - For each dimension col, we compute KL(hist_obs, hist_sim).
    - The total log-likelihood = - sum_of_KLs across all dimensions.
    """
    def __init__(self, simulator, param_names, bins=20):
        """
        simulator : object with set_params(list) and generate_samples(N)->(N,d)
        param_names : list of strings matching theta dimension
        bins : int or sequence, bin specification for np.histogram
        """
        self.simulator = simulator
        self.param_names = param_names
        self.bins = bins

    def log_likelihood(self, theta: np.ndarray, data: np.ndarray) -> float:
        self.simulator.set_params(theta)

        # 1) Generate simulated data with same number of samples as observed
        sim_data = self.simulator.generate_samples(len(data))
        # shapes: data -> (N, d), sim_data -> (N, d)

        # 2) For each dimension, compute 1D KL
        eps = 1e-8
        total_kl = 0.0
        d = data.shape[1]
        for col in range(d):
            obs_col = data[:, col]
            sim_col = sim_data[:, col]

            obs_hist, _ = np.histogram(obs_col, bins=self.bins, density=True)
            sim_hist, _ = np.histogram(sim_col, bins=self.bins, density=True)

            # Avoid zeros
            obs_hist += eps
            sim_hist += eps

            col_kl = entropy(obs_hist, sim_hist)  # KL from obs -> sim
            total_kl += col_kl

        # 3) Return negative of sum of KLs
        return - total_kl
