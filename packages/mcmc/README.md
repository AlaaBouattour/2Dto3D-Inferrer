# MCMC

A simple Metropolis-Hastings MCMC package that allows you to define priors and likelihoods, with the goal of estimating the parameters of a given generator (see generator package).

**Overview**

This package provides:

- **mcmc.prior**: Classes related to prior distributions (e.g. Uniform).
- **mcmc.likelihood**: Classes related to likelihood functions (e.g. KDE or KL).
- **mcmc.mcmc_sampler**: A Metropolis-Hastings sampler (MCMCSampler) that combines the prior and likelihood and finds the most probable parameters of the given generator.

All classes are designed to be easily extended or replaced with custom implementations.

```
mcmc/
  ├── __init__.py
  ├── prior.py
  ├── likelihood.py
  ├── mcmc_sampler.py
  └── ...
```