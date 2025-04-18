import numpy as np

# =============================================================================
# Base class for all distributions
# =============================================================================

class Distribution:
    """
    Abstract base class for sampling.
    
    The instance is initialized with a parameter vector 'theta' (a list or array)
    containing independent parameters. Subclasses should interpret the components accordingly.
    """
    def __init__(self, theta):
        self.theta = theta

    def sample(self, size=3):
        """
        Must be implemented by subclasses.
        
        For "ordered" distributions, should return an array of length 'size'
        (sorted in descending order).
        """
        raise NotImplementedError("The sample method must be implemented by subclasses.")

    def set_params(self, theta):
        """
        Update the parameter vector.
        """
        self.theta = theta


# =============================================================================
# Ordered distributions for d-dimensional ellipsoids
# =============================================================================

class Uniform(Distribution):
    def __init__(self, theta):
        """
        Expects theta = [u, l] where:
          - u: lower bound,
          - l: length of the interval (must be non-negative).
        The upper bound is computed as v = u + l.
        """
        if theta[1] < 0:
            raise ValueError("Length parameter (l) must be non-negative.")
        super().__init__(theta)

    def sample(self, size=3):
        u, l = self.theta
        v = u + l
        # Draw 'size' independent samples from U[u, v]
        values = np.random.uniform(low=u, high=v, size=size)
        sorted_vals = np.sort(values)[::-1]
        return sorted_vals  # returns an array of length 'size'


class Beta(Distribution):
    def __init__(self, theta):
        """
        Expects theta = [alpha, beta, scale_u, l] where:
          - alpha, beta: shape parameters for the Beta distribution,
          - scale_u: lower bound for scaling,
          - l: length of the scaling interval (must be non-negative).
        The upper bound is computed as scale_v = scale_u + l.
        """
        if theta[3] < 0:
            raise ValueError("Length parameter (l) must be non-negative.")
        super().__init__(theta)

    def sample(self, size=3):
        alpha, beta_val, scale_u, l = self.theta
        scale_v = scale_u + l
        values = np.random.beta(alpha, beta_val, size=size)
        scaled = scale_u + (scale_v - scale_u) * values
        sorted_vals = np.sort(scaled)[::-1]
        return sorted_vals


class Dirac(Distribution):
    def __init__(self, theta):
        """
        Expects theta to be a list of fixed values, e.g. [a, b, c] (or of any length) in descending order.
        Always returns the same vector, sorted in descending order.
        """
        super().__init__(theta)

    def sample(self, size=None):
        if size is None:
            size = len(self.theta)
        if size != len(self.theta):
            raise ValueError("For DiracDistribution, the requested size must equal the length of theta.")
        sorted_vals = np.sort(self.theta)[::-1]
        return np.array(sorted_vals)



    
