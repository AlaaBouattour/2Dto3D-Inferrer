import numpy as np
import matplotlib.pyplot as plt

from .prior import Prior
from .likelihood import Likelihood

class Sampler:
    """
    Generic Metropolis-Hastings MCMC over a theta of any dimension.
       
    - The acceptance step follows the standard Metropolis rule:
    
      Given a proposed state θ_prop and current state θ_current, the acceptance 
      probability is:
      
          alpha = min(1, p(θ_prop) / p(θ_current))

      Since we work in log-space to avoid numerical underflows, we rewrite this as:
      
          log_alpha = log p(θ_prop) - log p(θ_current)
      
      To make the acceptance decision, we compare `log_alpha` with the log of a 
      uniform random number in (0,1):

          log(U(0,1)) ~ (-∞, 0]    # log of a uniform number is always negative

      Thus, the decision:

          log(U(0,1)) < log_alpha   ie   U(0,1) < exp(log_alpha)

      This is mathematically equivalent to the original Metropolis acceptance rule.
      
    - Developers extending this to other distributions should ensure:
        1. The log-posterior (`log_posterior`) is computed correctly.
        2. The proposal mechanism matches the target distribution.
        3. The acceptance condition remains valid in log-space.
    """
    def __init__(self, prior: Prior, likelihood: Likelihood, data: np.ndarray):
        self.prior = prior
        self.likelihood = likelihood
        self.data = data
        self.chain = None

        # attribute to store the number of accepted proposals
        self.n_accepted = 0

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.prior.log_prior(theta)
        if np.isinf(lp):
            return lp
        ll = self.likelihood.log_likelihood(theta, self.data)
        return lp + ll

    def metropolis_hastings(self, init_theta, n_iter=2000, proposal_scale=0.1):
        """
        Perform Metropolis-Hastings sampling.
        
        Parameters
        ----------
        init_theta : np.ndarray (dim,)
            Initial parameter guess
        n_iter : int
            Number of iterations
        proposal_scale : float or np.ndarray
            Std dev(s) of proposal distribution (normal)
        
        Returns
        -------
        chain : np.ndarray, shape (n_iter, dim+1)
            Each row is [theta..., log_post].
        """
        dim = len(init_theta)
        current_theta = np.array(init_theta, dtype=float)
        current_logpost = self.log_posterior(current_theta)

        # Reset accepted count each time we run MCMC
        self.n_accepted = 0

        chain = []
        for _ in range(n_iter):
            # 1) Propose a new theta
            proposal = current_theta + np.random.normal(0, proposal_scale, size=dim)
            prop_logpost = self.log_posterior(proposal)

            # 2) Accept/Reject
            log_alpha = prop_logpost - current_logpost
            if np.log(np.random.rand()) < log_alpha:
                current_theta = proposal
                current_logpost = prop_logpost
                self.n_accepted += 1  # increment accepted proposals

            chain.append(np.concatenate([current_theta, [current_logpost]]))

        self.chain = np.array(chain)
        return self.chain

    def acceptance_rate(self):
        """
        Returns the fraction of proposals that were accepted in the last run.
        """
        if self.chain is None:
            raise ValueError("No chain found. Run metropolis_hastings first.")
        n_iter = len(self.chain)
        return self.n_accepted / n_iter

    def plot_trace(self, burn_in=0, true_params=None):
        """
        Plot a trace plot for each dimension of theta.
        """
        if self.chain is None:
            raise ValueError("No chain found. Run metropolis_hastings first.")
        chain_post = self.chain[burn_in:]
        dim = chain_post.shape[1] - 1  # last column is log_posterior

        fig, axes = plt.subplots(dim, 1, figsize=(8, 3*dim), sharex=True)
        if dim == 1:
            axes = [axes]

        for i in range(dim):
            axes[i].plot(chain_post[:, i], label=f"Param {i}")
            if true_params is not None and len(true_params) == dim:
                axes[i].axhline(true_params[i], color='r', linestyle='--', label='True')
            axes[i].legend()

        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        plt.show()

    def plot_hist(self, burn_in=0, bins=30, true_params=None):
        """
        Plot a histogram for each dimension of theta (posterior distribution).
        """
        if self.chain is None:
            raise ValueError("No chain found. Run metropolis_hastings first.")
        chain_post = self.chain[burn_in:]
        dim = chain_post.shape[1] - 1

        fig, axes = plt.subplots(1, dim, figsize=(4*dim, 4))
        if dim == 1:
            axes = [axes]

        for i in range(dim):
            axes[i].hist(chain_post[:, i], bins=bins, alpha=0.7)
            axes[i].set_title(f"Param {i}")
            if true_params is not None and len(true_params) == dim:
                axes[i].axvline(true_params[i], color='r', linestyle='--', label='True')
                axes[i].legend()

        plt.tight_layout()
        plt.show()
