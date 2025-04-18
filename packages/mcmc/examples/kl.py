import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import granulometre

import mcmc.prior as prior
import mcmc.likelihood as likelihood
import mcmc

np.random.seed(42)

##############################################################################
# 1) Generate synthetic data from "true" (u, l)
##############################################################################
#    If you want the final interval [1.2, 2.5], then l = 2.5 - 1.2 = 1.3
true_params = [1.2, 1.3]

dist_data = distributions.Uniform(true_params)
dist_data.set_params(true_params)

sim_data_gen = GranulometreSimulator(dist_data)
data_obs = sim_data_gen.generate_samples(100)
data_obs_df = pd.DataFrame(data_obs, columns=["Dmax", "Dmin"])

##############################################################################
# 2) Define a prior: 0 <= u < u + l <= 10
##############################################################################
prior_uniform = prior.Uniform(bounds=[(0,10),(0,10)], enforce_order=False)

##############################################################################
# 3) Define the simulator for inference
##############################################################################
theta_inference = [0,10]
dist_inference = distributions.Uniform(theta_inference)
sim_inference = GranulometreSimulator(dist_inference)

##############################################################################
# 4) Define a likelihood using KL divergence:
##############################################################################
#   choose bins=20 (or any integer).
param_names = ["u", "l"]
likelihood_KL = likelihood.KL(simulator=sim_inference, param_names=param_names, bins=20)

##############################################################################
# 5) Build MCMC sampler
##############################################################################
sampler = MCMCSampler(prior=prior_uniform, likelihood=likelihood_KL, data=data_obs_df.values)

##############################################################################
# 6) Run MCMC
##############################################################################
init_theta = [1.0, 2.0]  # example initial guess for (u, l)
n_iter = 3000

chain = sampler.metropolis_hastings(init_theta=init_theta, n_iter=n_iter, proposal_scale=0.1)

##############################################################################
# 7) Discard burn-in and plot
##############################################################################
burn_in = int(0.4 * n_iter)
sampler.plot_trace(burn_in=burn_in, true_params=true_params)
sampler.plot_hist(burn_in=burn_in, bins=30, true_params=true_params)

chain_post = chain[burn_in:]
u_mean = np.mean(chain_post[:, 0])
l_mean = np.mean(chain_post[:, 1])
print(f"KL approach -> Posterior mean u: {u_mean:.3f}, l: {l_mean:.3f}")
