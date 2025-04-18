import numpy as np

# =============================================================================
# Prior distributions for parameters (sampling candidate theta = (u, l))
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

class Uniform(Distribution):
    def __init__(self, theta):
        """
        Expects theta = [u_min, u_len, l_min, l_len], where:
          - u will be sampled uniformly in [u_min, u_min + u_len],
          - l will be sampled uniformly in [l_min, l_min + l_len].
          
        Checks that u_len and l_len are non-negative.
        """
        u_min, u_len, l_min, l_len = theta
        if u_len < 0:
            raise ValueError(f"PriorUniform init: u_len ({u_len}) must be non-negative.")
        if l_len < 0:
            raise ValueError(f"PriorUniform init: l_len ({l_len}) must be non-negative.")
        super().__init__(theta)

    def sample(self):
        u_min, u_len, l_min, l_len = self.theta
        u_val = np.random.uniform(low=u_min, high=u_min + u_len)
        l_val = np.random.uniform(low=l_min, high=l_min + l_len)
        # Return an array [u, l] (where v would be computed as u+l if needed)
        return np.array([u_val, l_val])


class Beta(Distribution):
    def __init__(self, theta):
        """
        Expects theta = [alpha_u, beta_u, u_min, u_len, alpha_l, beta_l, l_min, l_len], where:
          - u_raw ~ Beta(alpha_u, beta_u) is rescaled to [u_min, u_min + u_len],
          - l_raw ~ Beta(alpha_l, beta_l) is rescaled to [l_min, l_min + l_len].
          
        Checks that u_len and l_len are non-negative.
        """
        _, _, u_min, u_len, _, _, l_min, l_len = theta
        if u_len < 0:
            raise ValueError(f"PriorBeta init: u_len ({u_len}) must be non-negative.")
        if l_len < 0:
            raise ValueError(f"PriorBeta init: l_len ({l_len}) must be non-negative.")
        super().__init__(theta)

    def sample(self):
        alpha_u, beta_u, u_min, u_len, alpha_l, beta_l, l_min, l_len = self.theta
        
        raw_u = np.random.beta(alpha_u, beta_u)
        u_val = u_min + raw_u * u_len
        
        raw_l = np.random.beta(alpha_l, beta_l)
        l_val = l_min + raw_l * l_len
        
        return np.array([u_val, l_val])
    