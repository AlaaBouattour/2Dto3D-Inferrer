import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import granulometre

import mcmc.prior as prior
import mcmc.likelihood as likelihood
import mcmc

np.random.seed(42)

##############################################################################
# 1) Define MixtureOfUniforms class with the same [u, l] logic as "Uniform"
##############################################################################
class MixtureOfUniforms(granulometre.distribution.Distribution):
    """
    A mixture of two Uniform intervals for axis lengths:
      First uniform:  [u, u + l]
      Second uniform: [u2, u2 + l2]
    Probability alpha of picking the first uniform for a given (a,b,c),
    else we pick the second.

    theta = [u, l, u2, l2, alpha]
    """
    def __init__(self, theta):
        """
        Expects theta = [u, l, u2, l2, alpha], with l, l2 >= 0, 0 <= alpha <= 1.
        """
        super().__init__(theta)
    
    def sample(self, size=3):

        u, l, u2, l2, alpha = self.theta
        v  = u  + l
        v2 = u2 + l2
        
        # With probability alpha, draw from [u, v]; else from [u2, v2]
        if np.random.rand() < alpha:
            arr = np.random.uniform(low=u,  high=v,  size=size)
        else:
            arr = np.random.uniform(low=u2, high=v2, size=size)
        
        sorted_vals = np.sort(arr)[::-1]  # e.g. ensure a >= b >= c
        return sorted_vals

##############################################################################
# 2) Generate "observed" data from a true mixture
##############################################################################
# Suppose the "true" mixture parameters are:
#   u=1.0, l=1.5 => first interval is [1.0, 2.5]
#   u2=5.0, l2=3.0 => second interval is [5.0, 8.0]
#   alpha=0.4 => 40% from the first interval
true_params = [1.0, 1.5, 5.0, 3.0, 0.4]

dist_data = MixtureOfUniforms(true_params)
sim_data_gen = granulometre.Simulator(dist_data)

N = 200
data_obs = sim_data_gen.generate_samples(N)
data_obs_df = pd.DataFrame(data_obs, columns=["Dmax", "Dmin"])

##############################################################################
# 3) Define a 5D prior
##############################################################################
prior_uniform_5d = prior.Uniform(
    bounds=[
        (0,10),  # u
        (0,10),  # l
        (0,10),  # u2
        (0,10),  # l2
        (0,1)    # alpha
    ],
    enforce_order=False
)

##############################################################################
# 4) Define a mixture distribution
##############################################################################
theta_inference = [0.0, 2.0, 4.0, 3.0, 0.5]  # initial guess / placeholder
dist_inference = MixtureOfUniforms(theta_inference)
sim_inference  = granulometre.Simulator(dist_inference)

##############################################################################
# 5) Define the likelihood (KDE)
##############################################################################
param_names = ["u", "l", "u2", "l2", "alpha"]

likelihood_KDE = likelihood.KDE(
    simulator=sim_inference,
    param_names=param_names,
    bandwidth=0.3,
    n_sim=2000
)

##############################################################################
# 6) Build MCMC Sampler
##############################################################################
sampler = mcmc.Sampler(
    prior=prior_uniform_5d,
    likelihood=likelihood_KDE,
    data=data_obs_df.values
)

##############################################################################
# 7) Run MCMC
##############################################################################
init_theta = [0.5, 2.0, 5.5, 2.5, 0.3] 
n_iter = 3000

chain = sampler.metropolis_hastings(
    init_theta=init_theta,
    n_iter=n_iter,
    proposal_scale=0.1
)

##############################################################################
# 8) Discard burn-in and plot
##############################################################################
burn_in = int(0.4 * n_iter)
sampler.plot_trace(burn_in=burn_in, true_params=true_params)
sampler.plot_hist(burn_in=burn_in, bins=30, true_params=true_params)

chain_post = chain[burn_in:]
u_mean     = np.mean(chain_post[:, 0])
l_mean     = np.mean(chain_post[:, 1])
u2_mean    = np.mean(chain_post[:, 2])
l2_mean    = np.mean(chain_post[:, 3])
alpha_mean = np.mean(chain_post[:, 4])

print("\nPosterior means:")
print(f"   u:     {u_mean:.3f}")
print(f"   l:     {l_mean:.3f}")
print(f"   u2:    {u2_mean:.3f}")
print(f"   l2:    {l2_mean:.3f}")
print(f"   alpha: {alpha_mean:.3f}")