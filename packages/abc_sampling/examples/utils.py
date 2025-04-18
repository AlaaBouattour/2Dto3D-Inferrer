import numpy as np

class GaussianSimulator:
    """
    A simple simulator that generates data from a Gaussian (Normal) distribution.
    
    Expected parameter dictionary:
      {
         "mean": <float>,  # mean of the distribution
         "std": <float>    # standard deviation of the distribution
      }
    
    Methods:
      - set_params(new_params): Update the parameters.
      - generate_samples(N): Generate N data points from the Gaussian distribution.
    """
    def __init__(self, params):
        self.params = params  # e.g., mean= 0, std= 1 if params = [0, 1]

    def set_params(self, new_params):
        """
        Dynamically update the distribution parameters.
        """
        self.params = new_params

    def generate_samples(self, N):
        """
        Generate N data points from the Gaussian distribution.
        
        Returns:
          A numpy array of shape (N,)
        """
        mean = self.params[0]
        std = self.params[1]
        return np.random.normal(loc=mean, scale=std, size=N).reshape(N, 1)