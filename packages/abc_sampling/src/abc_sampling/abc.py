import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# RejectionSampling: Sampling and History Storage
# =============================================================================

class RejectionSampling:
    """
    Standard Rejection-based sampling approach (in an ABC-like scenario).

    Attributes:
      simulator: an instance of GranulometreSimulator.
      dataAB: a (M, d) numpy array representing the observed dataset of size M, containing elements in R^d.
      prior_sampler: an object inheriting from distributions.Distribution (with a .sample() method).
      distance_func: a function that computes the distance between simulated and observed data.
      
      accepted_params: list of candidate parameters (as arrays) that are accepted (distance < epsilon).
      history: list of arrays, each of shape (d+1,), where the first d elements are the candidate parameters
               and the last element is the computed distance.
      best_dist: the smallest distance found.
    """
    def __init__(self, simulator, dataAB, prior_sampler, distance_func):
        self.simulator = simulator
        self.dataAB = dataAB
        self.prior_sampler = prior_sampler
        self.distance_func = distance_func
        
        self.accepted_params = []  # List of candidate theta (numpy arrays) accepted.
        self.history = []          # Each entry: [theta_1, ..., theta_d, distance]
        self.best_dist = np.inf

    def run(self, N_iter=10000, epsilon=0.1):
        """
        Runs the Rejection-based sampling.

        Args:
          N_iter: number of iterations (parameter draws).
          epsilon: acceptance threshold for the distance.
        """
        M = self.dataAB.shape[0]
        
        for _ in tqdm(range(N_iter), desc="Rejection Sampling"):
            # 1) Sample candidate parameters (theta as numpy array)
            theta_candidate = self.prior_sampler.sample()
            
            # 2) Update the simulator with these parameters
            self.simulator.set_params(theta_candidate)
            
            # 3) Generate a simulated sample of size M
            simAB = self.simulator.generate_samples(M)
            
            # 4) Compute the distance
            dist_val = self.distance_func(simAB, self.dataAB)
            
            # 5) Record candidate parameters + distance
            entry = np.append(theta_candidate, dist_val)
            self.history.append(entry)
            
            # 6) Acceptance test
            if dist_val < epsilon:
                self.accepted_params.append(theta_candidate)
            
            # 7) Update best distance
            if dist_val < self.best_dist:
                self.best_dist = dist_val
        
        print(f"RejectionSampling finished. Best distance = {self.best_dist:.4f}")
        print(f"Number of accepted parameters = {len(self.accepted_params)}")
        
    def get_history(self):
        """
        Returns the history as a numpy array of shape (N_iter, d+1).
        """
        return np.array(self.history)
    
    def get_accepted_params(self):
        """
        Returns the list of accepted parameter arrays.
        """
        return self.accepted_params

# =============================================================================
# Interpolator: Fit a generic scikit-learn regressor
# =============================================================================

class Interpolator:
    """
    A generic interpolator that fits a scikit-learn regressor to approximate the
    mapping theta -> distance, using the (theta, distance) pairs in the history.
    """
    def __init__(self, history, regressor):
        """
        Args:
          history: a numpy array of shape (N, d+1), where the first d columns are theta,
                   and the last column is distance.
          regressor: any scikit-learn regressor implementing fit(X, y) and predict(X).
                     e.g. SVR(), RandomForestRegressor(), MLPRegressor(), etc.
        """
        if history.size == 0:
            raise ValueError("History is empty. Run the sampling before interpolating.")
        self.history = history
        self.regressor = regressor
        self.fitted = False

    def fit(self):
        """
        Fits the provided regressor to predict distance from theta.

        Returns:
          The trained regressor (same as self.regressor).
        """
        X = self.history[:, :-1]  # candidate theta vectors
        y = self.history[:, -1]   # distances

        self.regressor.fit(X, y)
        self.fitted = True
        return self.regressor

    def predict(self, X):
        """
        Predicts the distance for new parameter vectors X using the trained regressor.

        Args:
          X: numpy array of shape (n_samples, d), where d is dimension of theta.

        Returns:
          A numpy array of predicted distances.
        """
        if not self.fitted:
            raise RuntimeError("Regressor not fitted yet. Call fit() before predict().")
        return self.regressor.predict(X)

# =============================================================================
# HistoryPlotter: Plot the interpolated surface mapping theta -> distance
# =============================================================================

class HistoryPlotter:
    """
    Provides plotting utilities for the sampling results.
    
    In the case theta is 2-dimensional (e.g., [u, l]), we can plot a 3D surface
    of the predicted distance.
    """
    def __init__(self, history, model):
        """
        Args:
          history: numpy array of shape (N, 3) where columns are [theta1, theta2, distance].
          model: a trained regressor (e.g. from Interpolator) that maps theta -> distance.
        """
        if history.size == 0:
            raise ValueError("History is empty. Cannot plot.")
        self.history = history
        self.model = model

    def plot_history(self):
        """
        Plots a 3D scatter plot of (theta1, theta2, distance) from the history.
        """
        # For 2D thetas, we assume shape is (N, 3).
        u_vals = self.history[:, 0]
        l_vals = self.history[:, 1]
        d_vals = self.history[:, 2]
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(u_vals, l_vals, d_vals, c=d_vals, cmap='viridis', alpha=0.7)
        ax.set_xlabel("theta1")
        ax.set_ylabel("theta2")
        ax.set_zlabel("distance")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Distance")
        plt.title("Sampling History: theta -> distance")
        plt.show()

    def plot_interpolated_surface(self, grid_size=30):
        """
        Plots a 3D surface for the interpolated distance function f(theta) using the model.
        Assumes theta is 2D.
        
        Args:
          grid_size: number of points along each axis for creating the mesh grid.
        """
        all_data = np.array(self.history)
        theta1_vals = all_data[:, 0]
        theta2_vals = all_data[:, 1]
        dist_vals = all_data[:, 2]
        
        t1_min, t1_max = theta1_vals.min(), theta1_vals.max()
        t2_min, t2_max = theta2_vals.min(), theta2_vals.max()
        
        t1_grid = np.linspace(t1_min, t1_max, grid_size)
        t2_grid = np.linspace(t2_min, t2_max, grid_size)
        T1, T2 = np.meshgrid(t1_grid, t2_grid)
        T12_points = np.column_stack((T1.ravel(), T2.ravel()))
        
        dist_pred = self.model.predict(T12_points)
        Dist_grid = dist_pred.reshape(T1.shape)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        
        surf = ax.plot_surface(T1, T2, Dist_grid, cmap='viridis', alpha=0.7, edgecolor='none')
        sc = ax.scatter(theta1_vals, theta2_vals, dist_vals, color='red', s=1, label="Sampling History", depthshade=True)
        
        ax.set_xlabel("theta1")
        ax.set_ylabel("theta2")
        ax.set_zlabel("distance")
        plt.title("Interpolated Distance Surface with History")
        fig.colorbar(surf, ax=ax, shrink=0.6, label="Predicted Distance")
        plt.legend()
        plt.show()
